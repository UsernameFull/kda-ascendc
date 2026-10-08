"""#4, step 1: can a later stage's blocks run while an earlier one holds the
machine?

The cross-stage pipelining idea (docs 11.61's "next steps" item 4) needs one
fact before any slice machinery is built: when stage N runs, do stages N+1's
blocks get cores, or are the cores held for the whole stage?  The occupied
resource is not a pipe but a *core slot*: a MIX block holds 1 AIC + 2 AIV
until its program ends even if its pipes idle inside, an AIV-only block holds
AIV cores, an AIC-only block holds AIC cores, and a core runs one block at a
time.  Under that model the only cross-stage pair with free cores is AIV-only
beside AIC-only (which the two-level solve already ships internally), and
every MIX-vs-anything overlap is zero-sum.

This probe replays the production launches (captured once from a real call,
same pointers, same kernels, same shapes) in arms:

  p     pre_gram alone (one MIX launch)            -> its own span
  w     wide slices alone (AIV-only)               -> its own span
  ac    assemble + cube slices alone (AIC-only)    -> its own span
  k2    the persistent K2 alone (one MIX launch)   -> its own span
  solve the two-level solve as shipped (w on one stream, per-slice event to
        the ac stream)                            -> positive control
  p|ac  pre_gram and AIC-only solve work concurrently on two streams
  p|w   pre_gram and the AIV-only wide concurrently (P-to-S, AIV half)
  p|k2  pre_gram and K2 concurrently
  w|k2  wide and K2 concurrently
  ac|k2 assemble+cube and K2 concurrently
  solve|k2  the shipped solve schedule with K2 beside it (the S-to-K2
        ceiling, ungated)

The p|ac / p|w arms are the P-to-S question (both halves); w|k2 / ac|k2 /
solve|k2 are the S-to-K2 question piece by piece.  The arms ignore
data dependencies (the replay is for *scheduling*: whether cores are shared),
so outputs are meaningless and only spans are read; every arm is reported as
MIN of KDA_CORES_ROUNDS with the arm order rotated per round.

Predictions: under the occupancy model, p|ac ~ p + ac, p|k2 ~ p + k2,
w|k2 ~ w + k2, and the control reads ~ solve's own wall; if instead kernels
share cores at the pipe level, the concurrent arms come in well under the sum.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_stage_coresidency.py
"""
from __future__ import annotations

import faulthandler
import os
import sys
import time
from pathlib import Path

import torch
import torch_npu

faulthandler.dump_traceback_later(7200, exit=True)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import kda_ascendc_v1.api as api  # noqa: E402

D, DEV = 128, torch.device("npu:0")
B, T, H = 1, 8192, 96
ROUNDS = int(os.environ.get("KDA_CORES_ROUNDS", "8"))
P, W, A, C, K2 = ("kda_pre_gram_mix", "kda_solve_wu_wide",
                  "kda_solve_assemble", "kda_solve_wu_cube_kernel",
                  "kda_k2_persistent_loop")


def inputs():
    torch.manual_seed(1312)
    q = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    k = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    v = (torch.randn(B, T, H, D, device=DEV) * 0.1).to(torch.bfloat16)
    g = torch.randn(B, T, H, D, device=DEV) * 0.1
    beta = torch.randn(B, T, H, device=DEV)
    kw = dict(A_log=torch.linspace(-1.0, 0.2, H, device=DEV),
              bias=torch.randn(H, D, device=DEV) * 0.03,
              lower_bound=-1.0, output_final_state=True)
    return q, k, v, g, beta, kw


def capture(q, k, v, g, beta, kw):
    """One production call, launches recorded verbatim (args are reused)."""
    rec = {name: [] for name in (P, W, A, C, K2)}
    orig = api._launch

    def spy(kernel, blocks, args, stream):
        if kernel in rec:
            rec[kernel].append((kernel, int(blocks), list(args)))
        return orig(kernel, blocks, args, stream)

    api._launch = spy
    try:
        out, st = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    finally:
        api._launch = orig
    torch.npu.synchronize()
    for name in rec:
        print("captured %-28s x%d" % (name, len(rec[name])), flush=True)
    return rec


def ac_interleaved(rec):
    """Production order of the AIC-only half: asm_i then cube_i per slice."""
    out = []
    for asm, cube in zip(rec[A], rec[C]):
        out.append(asm)
        out.append(cube)
    return out


def reissue(launches, stream_raw):
    for name, blocks, args in launches:
        api.launch_argsarray_engine(name, blocks, stream_raw, args, 0)


def timed_arms(rec, rounds):
    """Every arm: events bracket each stream's sequence; span = the arm's
    first-start..last-end window, wall = host time around the enqueues +
    device sync."""
    cur = torch_npu.npu.current_stream()
    sa, sb = api._solve_streams(DEV)
    s3 = torch_npu.npu.Stream(device=DEV)
    streams = {"cur": cur, "sa": sa, "s3": s3}
    plan = {
        "p":      [("cur", rec[P])],
        "w":      [("cur", rec[W])],
        "ac":     [("cur", ac_interleaved(rec))],
        "k2":     [("cur", rec[K2])],
        "p|ac":   [("cur", rec[P]), ("sa", ac_interleaved(rec))],
        "p|w":    [("cur", rec[P]), ("sa", rec[W])],
        "p|k2":   [("cur", rec[P]), ("sa", rec[K2])],
        "w|k2":   [("cur", rec[W]), ("sa", rec[K2])],
        "ac|k2":  [("cur", ac_interleaved(rec)), ("sa", rec[K2])],
    }
    names = list(plan) + ["solve", "solve|k2"]
    res = {n: [] for n in names}
    for r in range(rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        for n in order:
            t0 = time.perf_counter()
            if n == "solve":
                span = _solve_control(rec, sa, sb)
            elif n == "solve|k2":
                span = _solve_control(rec, sa, sb, k2_stream=s3)
            else:
                starts, ends = [], []
                for key, launches in plan[n]:
                    st = torch_npu.npu.Event(enable_timing=True)
                    en = torch_npu.npu.Event(enable_timing=True)
                    st.record(streams[key])
                    reissue(launches, streams[key].npu_stream)
                    en.record(streams[key])
                    starts.append(st)
                    ends.append(en)
                torch.npu.synchronize()
                ref = starts[0]
                rel = [e.elapsed_time(ref) for e in starts + ends]
                span = max(rel) - min(rel)
            wall = (time.perf_counter() - t0) * 1e3
            res[n].append((wall, span))
            torch.npu.synchronize()
    return res


def _solve_control(rec, sa, sb, k2_stream=None):
    """The shipped two-level solve schedule: wide on one stream, a per-slice
    event hands each slice's xb/lneg to the asm+cube stream.  With
    ``k2_stream`` the K2 launch runs beside the schedule (ungated: the S-to-K2
    co-residency ceiling, not a legal pipeline)."""
    ev_s = torch_npu.npu.Event(enable_timing=True)
    ev_s.record(sa)
    ek0 = ek1 = None
    if k2_stream is not None:
        ek0 = torch_npu.npu.Event(enable_timing=True)
        ek1 = torch_npu.npu.Event(enable_timing=True)
        ek0.record(k2_stream)
        reissue(rec[K2], k2_stream.npu_stream)
        ek1.record(k2_stream)
    for wide, asm, cube in zip(rec[W], rec[A], rec[C]):
        reissue([wide], sa.npu_stream)
        ev = torch_npu.npu.Event()
        ev.record(sa)
        sb.wait_event(ev)
        reissue([asm, cube], sb.npu_stream)
    ev_e = torch_npu.npu.Event(enable_timing=True)
    ev_e.record(sb)
    torch.npu.synchronize()
    if k2_stream is None:
        return ev_s.elapsed_time(ev_e)
    rel = [ev_e.elapsed_time(ev_s), ek0.elapsed_time(ev_s), ek1.elapsed_time(ev_s)]
    return max(rel) - min(0.0, min(rel))


def main() -> None:
    q, k, v, g, beta, kw = inputs()
    rec = capture(q, k, v, g, beta, kw)
    res = timed_arms(rec, ROUNDS)
    mins = {n: (min(w for w, _ in res[n]), min(s for _, s in res[n]))
            for n in res}
    print("\nspans (MIN of %d), wall / span" % ROUNDS, flush=True)
    for n in ("p", "w", "ac", "k2", "solve", "p|ac", "p|w", "p|k2",
              "w|k2", "ac|k2", "solve|k2"):
        w, s = mins[n]
        print("  %-6s wall %.3f   span %.3f" % (n, w, s), flush=True)
    p, w, ac, k2 = (mins["p"][1], mins["w"][1], mins["ac"][1], mins["k2"][1])
    sv = mins["solve"][1]
    print("\n  predictions if cores are held (sum) vs shared (max):", flush=True)
    for n, a, b in (("p|ac", p, ac), ("p|w", p, w), ("p|k2", p, k2),
                    ("w|k2", w, k2), ("ac|k2", ac, k2), ("solve|k2", sv, k2)):
        got = mins[n][1]
        print("  %-6s  sum %.3f  max %.3f  got %.3f  (vs sum %+.3f)"
              % (n, a + b, max(a, b), got, got - a - b), flush=True)
    print("  solve control: got %.3f (two halves w %.3f / ac %.3f)"
          % (sv, w, ac), flush=True)


if __name__ == "__main__":
    main()
