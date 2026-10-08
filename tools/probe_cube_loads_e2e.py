"""The Cube solve's band-merged RHS load in the production regime (docs 11.63).

The isolated replay (tools/probe_solve_cube_knobs.py) put the cube at
1.463 -> 0.996 ms with W/U bit-identical, which flips the solve's AIC floor
below the wide half (asm 0.47 + cube 1.53 = 2.00 against wide 1.85 before;
asm 0.47 + cube ~1.06 = ~1.53 after).  Whether the stage/e2e actually move is
the measurement this probe owns: same process, arms alternating every round.

  wall   host wall around the whole pipeline call (MIN of KDA_CUBE_ROUNDS)
  span   two device events bracketing exactly that call
  sums   one extra pass per arm with per-launch device events, summing the
         three solve kernels (wide / assemble / cube) and reading the
         first-to-last device span of the call
  sliced the solve schedule replayed out of one arm's capture (per-slice
         events, MIN of 5): the clean stage floor, free of host pacing

Identity on out/state for every arm against mode 0 from the same runs.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_cube_loads_e2e.py
  # KDA_CUBE_SKIP_SPANS=1 skips the per-launch-event pass (events make the run
  # host-paced, docs 11.59/11.61)
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
ROUNDS = int(os.environ.get("KDA_CUBE_ROUNDS", "12"))
ARMS = [int(x) for x in os.environ.get("KDA_CUBE_ARMS", "0,1").split(",")]
ENV = "KDA_CUBE_LOADS"
SOLVE = ("kda_solve_wu_wide", "kda_solve_assemble", "kda_solve_wu_cube_kernel")


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


def timed(q, k, v, g, beta, kw, mode):
    os.environ[ENV] = str(mode)
    cur = torch_npu.npu.current_stream()
    ev0 = torch_npu.npu.Event(enable_timing=True)
    ev1 = torch_npu.npu.Event(enable_timing=True)
    ev0.record(cur)
    t0 = time.perf_counter()
    out, st = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    ev1.record(cur)
    torch.npu.synchronize()
    return out, st, (time.perf_counter() - t0) * 1e3, ev0.elapsed_time(ev1)


def spans(q, k, v, g, beta, kw, mode):
    os.environ[ENV] = str(mode)
    cur = torch_npu.npu.current_stream()
    sa, sb = api._solve_streams(DEV)
    streams = {cur.npu_stream: cur, sa.npu_stream: sa, sb.npu_stream: sb}
    log = []
    orig = api._launch

    def spy(kernel, blocks, args, stream):
        obj = streams.get(stream)
        if obj is None:
            return orig(kernel, blocks, args, stream)
        e0 = torch_npu.npu.Event(enable_timing=True)
        e1 = torch_npu.npu.Event(enable_timing=True)
        e0.record(obj)
        ret = orig(kernel, blocks, args, stream)
        e1.record(obj)
        log.append((kernel, e0, e1))
        return ret

    api._launch = spy
    try:
        out, st = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    finally:
        api._launch = orig
    torch.npu.synchronize()
    tot = {}
    for name, e0, e1 in log:
        if name in SOLVE:
            tot[name] = tot.get(name, 0.0) + e0.elapsed_time(e1)
    ref = log[0][1]
    first = min((e0 for _, e0, _ in log), key=ref.elapsed_time)
    last = max((e1 for _, _, e1 in log), key=ref.elapsed_time)
    return out, st, tot, first.elapsed_time(last)


def _timeit(fn, reps=5):
    xs = []
    for _ in range(reps):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        xs.append((time.perf_counter() - t0) * 1e3)
    return min(xs)


def sliced_stage(q, k, v, g, beta, kw, mode):
    """The production solve schedule replayed: per-slice events, MIN of 5."""
    os.environ[ENV] = str(mode)
    seq = []
    orig = api._launch

    def spy2(kernel, blocks, args, stream):
        seq.append((kernel, int(blocks), list(args)))
        return orig(kernel, blocks, args, stream)

    api._launch = spy2
    try:
        api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    finally:
        api._launch = orig
    torch.npu.synchronize()
    units = [x for x in seq if x[0] in SOLVE]
    cur = torch_npu.npu.current_stream()
    sa, sb = api._solve_streams(DEV)

    def replay_sliced():
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

    return _timeit(replay_sliced), len(units)


def main() -> None:
    q, k, v, g, beta, kw = inputs()
    res = {a: [] for a in ARMS}
    outs = {}
    for r in range(ROUNDS):
        order = ARMS[r % len(ARMS):] + ARMS[:r % len(ARMS)]
        for a in order:
            out, st, wall, span = timed(q, k, v, g, beta, kw, a)
            res[a].append((wall, span))
            outs[a] = (out, st)
    base = min(w for w, _ in res[0]) if 0 in res else None
    print("pipeline wall / span (MIN of %d), arms = %s, env = %s"
          % (ROUNDS, ARMS, ENV), flush=True)
    for a in ARMS:
        w = min(x[0] for x in res[a])
        s = min(x[1] for x in res[a])
        print("  cube mode %d: wall %.3f%s  span %.3f%s"
              % (a, w, "" if base is None or a == 0 else " (%+.3f)" % (w - base),
                 s, "" if 0 not in res or a == 0 else
                 " (%+.3f)" % (s - min(x[1] for x in res[0]))), flush=True)
    print("spread of walls: %s"
          % ", ".join("m%d %.3f" % (a, max(x[0] for x in res[a])
                                    - min(x[0] for x in res[a]))
                      for a in ARMS), flush=True)
    ref_out, ref_st = outs[0]
    for a in ARMS:
        o, st = outs[a]
        print("  identity mode %d vs 0: out %d, state %d"
              % (a, int((o != ref_out).sum().item()),
                 int((st != ref_st).sum().item())), flush=True)
    if os.environ.get("KDA_CUBE_SKIP_SPANS"):
        print("(per-launch event pass skipped: events make the run host-paced, "
              "docs 11.59/11.61)", flush=True)
    else:
        print()
        for a in ARMS:
            o, st, tot, stage = spans(q, k, v, g, beta, kw, a)
            print("cube mode %d device sums: wide %.3f | asm %.3f | cube %.3f | "
                  "AIC %.3f | stage(first..last) %.3f"
                  % (a, tot.get(SOLVE[0], 0.0), tot.get(SOLVE[1], 0.0),
                     tot.get(SOLVE[2], 0.0),
                     tot.get(SOLVE[1], 0.0) + tot.get(SOLVE[2], 0.0), stage),
                  flush=True)
    if not os.environ.get("KDA_CUBE_SKIP_SLICED"):
        print()
        print("=== solve schedule replay (per-slice events, MIN of 5) ===")
        base = None
        for a in ARMS:
            t, n = sliced_stage(q, k, v, g, beta, kw, a)
            if base is None:
                base = t
            print("  cube mode %d: sliced stage %.3f ms (%+.3f) [%d launches]"
                  % (a, t, t - base, n), flush=True)


if __name__ == "__main__":
    main()
