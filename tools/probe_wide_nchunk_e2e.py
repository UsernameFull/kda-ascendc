"""The wide kernel's chunks-per-block knob (KDA_SOLVE_WIDE_NCHUNK) at e2e.

The isolated replay (tools/probe_solve_wide_nchunk.py) prices the kernel; this
probe prices the pipeline.  NCHUNK is a compile-time define baked into the wide
kernel at import, so unlike the cube-loads probe the arms cannot be flipped in
one process: run this script once per NCHUNK and interleave the invocations.

Four readings, same shapes as tools/probe_cube_loads_e2e.py:

  wall   host wall around the whole pipeline call (MIN of KDA_ROUNDS)
  span   two device events bracketing exactly that call
  sums   one extra pass with per-launch device events, summing the three solve
         kernels (wide / assemble / cube) and the first-to-last device span
  sliced the solve schedule replayed out of one pass (per-slice events, MIN of
         5): the clean stage floor, free of host pacing

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_wide_nchunk_e2e.py
"""
from __future__ import annotations

import faulthandler
import os
import sys
import time
from pathlib import Path

import torch
import torch_npu

faulthandler.dump_traceback_later(3600, exit=True)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import kda_ascendc_v1.api as api  # noqa: E402

D, DEV = 128, torch.device("npu:0")
B, T, H = 1, 8192, 96
ROUNDS = int(os.environ.get("KDA_ROUNDS", "8"))
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


def timed(q, k, v, g, beta, kw):
    cur = torch_npu.npu.current_stream()
    ev0 = torch_npu.npu.Event(enable_timing=True)
    ev1 = torch_npu.npu.Event(enable_timing=True)
    ev0.record(cur)
    t0 = time.perf_counter()
    out, st = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    ev1.record(cur)
    torch.npu.synchronize()
    return out, st, (time.perf_counter() - t0) * 1e3, ev0.elapsed_time(ev1)


def spans(q, k, v, g, beta, kw):
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


def sliced_stage(q, k, v, g, beta, kw):
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
    print("KDA_SOLVE_WIDE_NCHUNK = %d  (CHUNK %d, SUBB %d, NCH %d, "
          "SOLVE_OVERLAP %d, slice mode %s)"
          % (api.SOLVE_WIDE_NCHUNK, api.CHUNK, api.SOLVE_WIDE_SUBB,
             api.SOLVE_WIDE_NCH, api.SOLVE_OVERLAP,
             os.environ.get("KDA_SOLVE_SLICE_CHUNKS", "0")), flush=True)
    q, k, v, g, beta, kw = inputs()
    res = []
    for _ in range(ROUNDS):
        _, _, wall, span = timed(q, k, v, g, beta, kw)
        res.append((wall, span))
    print("pipeline wall %.3f  span %.3f  (MIN of %d; wall spread %.3f)"
          % (min(w for w, _ in res), min(s for _, s in res), ROUNDS,
             max(w for w, _ in res) - min(w for w, _ in res)), flush=True)
    o, st, tot, stage = spans(q, k, v, g, beta, kw)
    print("device sums: wide %.3f | asm %.3f | cube %.3f | AIC %.3f | "
          "stage(first..last) %.3f"
          % (tot.get(SOLVE[0], 0.0), tot.get(SOLVE[1], 0.0),
             tot.get(SOLVE[2], 0.0),
             tot.get(SOLVE[1], 0.0) + tot.get(SOLVE[2], 0.0), stage), flush=True)
    t, n = sliced_stage(q, k, v, g, beta, kw)
    print("sliced stage %.3f ms [%d launches]" % (t, n), flush=True)


if __name__ == "__main__":
    main()
