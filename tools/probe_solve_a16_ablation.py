"""The wide kernel's remaining DMA: what do its parent-tile stores cost?

Plan 11.33.2 measured the AIV half of the solve at 2.123 ms with a 1.169 ms
recursion floor, i.e. ~0.95 ms of gathers, casts and stores.  The order that
followed names the stores as the first thing to price, before touching the
recursion or the tile layout, and this probe does exactly that:

  mode 0  production: the parent A_inv tile gets its two diagonal sub-blocks
          (X11/X22, the bf16 round trip of the row recursion) and the
          strict-upper blank the Cube solve reads as zeros
  mode 1  no blank
  mode 2  no parent tile at all

Modes 1/2 leave `kda_solve_wu_cube_kernel` a partially written operand, so their
*outputs are wrong by construction* - they are ablations, not candidates, and
the arms are compared on time only.  That is not a figure of speech here: the
cube really does read the parent tile, and an ablation arm only *looks* correct
because the caching allocator hands the next call the same 98 MB block with the
previous call's contents still in it.  ``--cold`` measures that: it runs mode 2
as the very first call of the process (an A16 of uninitialised memory) and
compares it against the mode 0 that follows.  What makes them worth running is the size
of the answer: if the parent tile is not the 0.95 ms, then the remaining plan
(Xb/Lneg consumed in place, the round-trip casts) cannot be either, and the
solve stage should be left alone.

Everything happens in one process: the mode is a kernel *argument* read from
KDA_SOLVE_A16_MODE per call (no recompile between arms), so the e2e arm is the
real api call round-robined over modes and the isolated arms replay each mode's
own captured launch args.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=1 python3 -u tools/probe_solve_a16_ablation.py
"""
from __future__ import annotations

import faulthandler
import os
import sys
import time
from pathlib import Path

import torch
import torch_npu
from triton.testing import do_bench

faulthandler.dump_traceback_later(3600, exit=True)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import kda_ascendc_v1.api as api  # noqa: E402

D, DEV = 128, torch.device("npu:0")
B, T, H = 1, 8192, 96
MODES = [(0, "production (diag+blank)"), (1, "no blank"), (2, "no parent tile")]


def inputs():
    torch.manual_seed(1312)
    q = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    k = (torch.randn(B, T, H, D, device=DEV) * 0.2).to(torch.bfloat16)
    v = (torch.randn(B, T, H, D, device=DEV) * 0.1).to(torch.bfloat16)
    g = torch.randn(B, T, H, D, device=DEV) * 0.1
    beta = torch.randn(B, T, H, device=DEV)
    a_log = torch.linspace(-1.0, 0.2, H, device=DEV)
    bias = torch.randn(H, D, device=DEV) * 0.03
    return q, k, v, g, beta, a_log, bias


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold", action="store_true",
                    help="run mode 2 as the process's first call, to show that "
                         "the ablation only looks correct because A16 is stale")
    ap.add_argument("--arms-only", action="store_true",
                    help="skip the e2e sweep and the profiled section (the "
                         "replay arms carry the attribution); used when the run "
                         "is about a knob the e2e arm cannot resolve")
    ap.add_argument("--rev", action="store_true",
                    help="round-robin the e2e arms in reverse order, so a "
                         "systematic drift between the three positions shows up "
                         "as a sign flip")
    args = ap.parse_args()
    q, k, v, g, beta, a_log, bias = inputs()
    kw = dict(A_log=a_log, bias=bias, lower_bound=-1.0, output_final_state=True)

    def call(mode: int):
        os.environ["KDA_SOLVE_A16_MODE"] = str(mode)
        return api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)

    if args.cold:
        print("cold check: mode 2 first (RTC at C=%d) ..." % api.CHUNK, flush=True)
        cold_out, cold_st = call(2)
        torch.npu.synchronize()
        warm = call(0)
        torch.npu.synchronize()
        nd = int((cold_out != warm[0]).sum())
        print("   mode 2 on a fresh A16 vs mode 0: %d/%d elements differ, "
              "max|d| %.3e" % (nd, cold_out.numel(),
                               float((cold_out.float() - warm[0].float()).abs().max())),
              flush=True)
        del cold_out, cold_st, warm

    print("warming (RTC at C=%d) + identity check ..." % api.CHUNK, flush=True)
    ref = call(0)
    torch.npu.synchronize()
    again = call(0)
    torch.npu.synchronize()
    same = (torch.equal(ref[0], again[0]) and torch.equal(ref[1], again[1]))
    print("   mode 0 is deterministic: %s" % same, flush=True)
    for mode, name in MODES[1:]:
        out, st = call(mode)
        torch.npu.synchronize()
        diff = float((out.float() - ref[0].float()).abs().max())
        nd = int((out != ref[0]).sum())
        print("   mode %d (%s): %d/%d elements differ, max|d| %.3e   [stale A16 "
              "makes this read 0; see --cold]"
              % (mode, name, nd, out.numel(), diff), flush=True)

    if args.arms_only:
        print()
        print("(--arms-only: e2e and profile sections skipped)", flush=True)
        os.environ.pop("KDA_SOLVE_A16_MODE", None)
        return

    print()
    print("=== full pipeline, mode round-robin, do_bench median of 3 ===")
    med = {}
    order = list(reversed(MODES)) if args.rev else list(MODES)
    for rnd in range(3):
        for mode, name in order:
            m = do_bench(lambda md=mode: call(md), warmup=50, rep=400)
            med.setdefault(mode, []).append(m)
    base = None
    for mode, name in MODES:
        vals = sorted(med[mode])
        mid = vals[len(vals) // 2]
        if base is None:
            base = mid
        print("   mode %d %-22s %.3f ms (rounds %s)  Δ %+.3f ms"
              % (mode, name, mid, " ".join("%.3f" % x for x in sorted(med[mode])),
                 mid - base), flush=True)

    # ---- isolated solve arms, per mode, from that mode's own captured args ---
    print()
    print("=== isolated solve (replay of each mode's captured launches, MIN of 5) ===")
    cur = torch_npu.npu.current_stream()
    sa = torch_npu.npu.Stream(device=DEV)
    sb = torch_npu.npu.Stream(device=DEV)

    def timeit(fn, reps=5):
        xs = []
        for _ in range(reps):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.npu.synchronize()
            xs.append((time.perf_counter() - t0) * 1e3)
        return min(xs)

    print("   mode  AIV      ASM      CUBE     AIC      overlapped  sliced")
    for mode, name in MODES:
        seq = []
        orig = api._launch

        def spy(kn, blocks, args, stream, _seq=seq):
            _seq.append((kn, int(blocks), list(args)))
            return orig(kn, blocks, args, stream)

        api._launch = spy
        call(mode)
        torch.npu.synchronize()
        api._launch = orig

        def play(names, stream):
            for kn, blk, ar in seq:
                if kn in names:
                    api.launch_argsarray_engine(kn, blk, stream.npu_stream, ar, 0)

        def replay_sliced(units):
            """The production schedule, replayed: wide slice i on sa, an event,
            then that slice's assemble/cube on sb.  The event is what the plain
            "both" arm below leaves out, so this arm is the stage's real floor.
            ``units`` is the capture filtered to the solve's three kernels (the
            capture also holds pre_gram and K2, and the first launch of a call
            is never a wide one)."""
            sa.wait_stream(cur)
            sb.wait_stream(cur)
            i = 0
            while i < len(units):
                kn, blk, ar = units[i]
                if kn != "kda_solve_wu_wide":
                    i += 1
                    continue
                api.launch_argsarray_engine(kn, blk, sa.npu_stream, ar, 0)
                ev = torch_npu.npu.Event()
                ev.record(sa)
                sb.wait_event(ev)
                j = i + 1
                while j < len(units) and units[j][0] != "kda_solve_wu_wide":
                    k2, b2, a2 = units[j]
                    api.launch_argsarray_engine(k2, b2, sb.npu_stream, a2, 0)
                    j += 1
                i = j
            cur.wait_stream(sa)
            cur.wait_stream(sb)

        asm_set = {"kda_solve_assemble"}
        cube_set = {"kda_solve_wu_cube_kernel"}
        rest = asm_set | cube_set
        wide = {"kda_solve_wu_wide"}
        aiv = timeit(lambda: (sa.wait_stream(cur), play(wide, sa), cur.wait_stream(sa)))
        asm = timeit(lambda: (sb.wait_stream(cur), play(asm_set, sb), cur.wait_stream(sb)))
        cub = timeit(lambda: (sb.wait_stream(cur), play(cube_set, sb), cur.wait_stream(sb)))
        aic = timeit(lambda: (sb.wait_stream(cur), play(rest, sb), cur.wait_stream(sb)))
        both = timeit(lambda: (sa.wait_stream(cur), sb.wait_stream(cur),
                               play(wide, sa), play(rest, sb),
                               cur.wait_stream(sa), cur.wait_stream(sb)))
        units = [x for x in seq if x[0] in (wide | rest)]
        sliced = timeit(lambda: replay_sliced(units))
        print("   %-5d %7.3f  %7.3f  %7.3f  %7.3f  %7.3f   %7.3f  "
              "(launches: %d wide / %d asm / %d cube)"
              % (mode, aiv, asm, cub, aic, both, sliced,
                 sum(1 for x in seq if x[0] in wide),
                 sum(1 for x in seq if x[0] in asm_set),
                 sum(1 for x in seq if x[0] in cube_set)), flush=True)

    # ---- the real stage, profiled (the decision number is solve_ms here, not
    # the replay above: the replay has no per-slice events between the streams,
    # so its "overlapped" column is a floor rather than the stage).
    print()
    print("=== KDA_PROFILE=1 per mode, the real call (marks include their sync) ===")
    os.environ["KDA_PROFILE"] = "1"
    for mode, name in MODES:
        call(mode)
        torch.npu.synchronize()
        prof = dict(api.get_last_profile())
        print("   mode %d %-22s pre_gram %6.3f  solve %6.3f  k2 %6.3f  total %6.3f ms"
              % (mode, name, prof.get("pre_gram_ms", -1), prof.get("solve_ms", -1),
                 prof.get("k2_ms", -1), prof.get("total_ms", -1)), flush=True)
    os.environ.pop("KDA_PROFILE", None)
    os.environ.pop("KDA_SOLVE_A16_MODE", None)


if __name__ == "__main__":
    main()
