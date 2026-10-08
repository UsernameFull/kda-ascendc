"""The cube solve's MTE2 load shape (plan 11.63), measured in production.

The on-board account (msprof op PipeUtilization, archive
/data/models/Qwen3-4B/kda_msprof_20261008_cube) reads the shipped
``kda_solve_wu_cube_kernel`` block wall as 4.382 us, of which MTE2 is
3.271 us (74.6%): 16 Nd2Nz calls of 4 KB (the per-band RHS loop) plus 2 of
8 KB (A16) per block.  Section 11.46 priced the RHS read at 1.427 ms/GB and
separated 0.232 ms of the 16-call pattern as pure call overhead; section
11.37's assemble lesson was that ndNum-batched calls land bit-identically and
cheaper.  This probe is that question for the cube kernel.

  mode 0  shipped       KF calls per chunk-pass + per-chunk A16
  mode 1  band-merged   one Nd2Nz (ndNum = KF) per chunk-pass
  mode 2  block-merged  one call per pass over the block (ndNum = KF * nch)
                        plus one A16 call per block (ndNum = nch)

All modes run ``kernels/v1/k1_solve_cube_knobs_probe.cpp``, a transcription
of ``k1_solve_wu_cube.cpp`` that differs in the load calls and nothing else.
The probe first replays the production capture in each mode and checks W/U
are bit-identical, then times the isolated replay (MIN of N, arm order
rotated) with the unchanged capture as the in-context anchor.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_solve_cube_knobs.py
  # NC sweep: the same probe under KDA_WU_NCHUNK=1 (grid doubles, mode 2
  # degrades to the single-chunk form by its nch > 1 guard)
  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 KDA_WU_NCHUNK=1 \\
      python3 -u tools/probe_solve_cube_knobs.py
"""
from __future__ import annotations

import argparse
import faulthandler
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
CHUNKS = H * (T // api.CHUNK)
PROBE = "kda_solve_cube_knobs_probe"
MODES = [(0, "shipped      (KF calls/chunk-pass + A16 x nch)"),
         (1, "band-merged  (1 call/chunk-pass)"),
         (2, "block-merged (1 call/pass + A16 x 1)")]


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--skip-identity", action="store_true")
    args = ap.parse_args()
    q, k, v, g, beta, a_log, bias = inputs()
    kw = dict(A_log=a_log, bias=bias, lower_bound=-1.0, output_final_state=True)

    # ---- production reference + full launch capture -------------------------
    seq = []
    orig = api._launch

    def spy(name, blocks, a, stream):
        seq.append((name, int(blocks), list(a)))
        return orig(name, blocks, a, stream)

    api._launch = spy
    print("warming the production pipeline (RTC) ...", flush=True)
    _, _, dbg = api.kda_bt16_fwd_ascendc(q, k, v, g, beta,
                                         return_intermediates=True, **kw)
    torch.npu.synchronize()
    api._launch = orig
    cube = [x for x in seq if x[0] == "kda_solve_wu_cube_kernel"]
    if not cube:
        raise SystemExit("no kda_solve_wu_cube_kernel launch captured")
    Wt, Ut = dbg["W"], dbg["U"]
    W_ref, U_ref = Wt.clone(), Ut.clone()
    blocks = sum(b for _, b, _ in cube)
    print("captured %d cube launches, %d blocks total, CHUNK=%d NC=%d"
          % (len(cube), blocks, api.CHUNK, api.WU_NCHUNK), flush=True)

    print("compiling probe kernel ...", flush=True)
    api._rtc("kernels/v1/k1_solve_cube_knobs_probe.cpp", PROBE)
    cur = torch_npu.npu.current_stream()
    cur_h = cur.npu_stream

    def replay(mode=None):
        for kn, blk, a in cube:
            if mode is None:
                api._launch(kn, blk, a, cur_h)
            else:
                api._launch(PROBE, blk, a + [api._i(mode)], cur_h)

    def timeit(fn):
        xs = []
        for _ in range(args.reps):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.npu.synchronize()
            xs.append((time.perf_counter() - t0) * 1e3)
        return min(xs)

    # ---- identity: each arm replays the production capture ------------------
    if not args.skip_identity:
        print()
        print("=== identity: W/U vs production, all %d launches replayed per arm ==="
              % len(cube))
        for mode, name in MODES:
            replay(mode)
            torch.npu.synchronize()
            dW = int((Wt != W_ref).sum())
            dU = int((Ut != U_ref).sum())
            mW = float((Wt.float() - W_ref.float()).abs().max())
            mU = float((Ut.float() - U_ref.float()).abs().max())
            print("   mode %d %-46s W %8d/%d max|d| %.3e ; U %8d/%d max|d| %.3e"
                  % (mode, name, dW, Wt.numel(), mW, dU, Ut.numel(), mU),
                  flush=True)

    # ---- isolated replay timing, arms interleaved ---------------------------
    print()
    print("=== isolated replay, MIN of %d x %d rounds (order rotated) ==="
          % (args.reps, args.rounds))
    replay(0)
    torch.npu.synchronize()
    best = {m: 1e9 for m, _ in MODES}
    anchor = timeit(lambda: replay(None))
    order = list(MODES)
    for rnd in range(args.rounds):
        if rnd % 2:
            order = list(reversed(order))
        for mode, _ in order:
            best[mode] = min(best[mode], timeit(lambda m=mode: replay(m)))
    base = best[0]
    print("   production capture (unchanged)  %7.3f ms  %6.1f us/block" % (anchor, anchor * 1e3 / blocks))
    for mode, name in MODES:
        print("   mode %d %-46s %7.3f ms  %+.3f ms vs mode 0  %6.1f us/block"
              % (mode, name, best[mode], best[mode] - base, best[mode] * 1e3 / blocks),
              flush=True)


if __name__ == "__main__":
    main()
