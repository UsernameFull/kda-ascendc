"""What sets the assemble's Fixpipe time: per-call cost, or the store shape?

The on-board PipeUtilization collection (docs section 11.42, msopprof) measured
the coupling-block kernel at 1.812 us of Fixpipe busy per block - 49.6% of its
3.652 us wall - from 8 calls of 2 KB each (4 chunks x 2 passes), i.e. 226.5 ns
per call and only 8.9 GB/s.  This probe prices the two readings:

  mode 0  the production store structure: 4x NZ -> L1 (pass 0), then 4x
          row-major -> GM with dstStride = PC (pass 1)           = 8 calls
  mode 1  mode 0 with pass 1's four stores batched into ONE call via
          ndNum / srcNdStride / dstNdStride - the candidate; it has to be
          bit-identical to mode 0 before it is timed
  mode 2  8x NZ -> L1 (what pass 0's bytes cost per call)
  mode 3  8x row-major -> GM, dstStride = M (contiguous 2 KB blocks)
  mode 4  8x row-major -> GM, dstStride = PC (production rows)
  mode 5  no stores: the loads + Mmad floor

mode 3 vs 4 separates the row-stride premium from the per-call cost, and both
vs 5 give the per-call price of each destination pattern.  The arithmetic is
identical in every arm, so the deltas are the store forms.

The batched call's srcNdStride is a runtime argument (1 KB units of the L0C
source; the slots are 4 KB apart so 4 is the first guess) - the driver sweeps
it and only times the value that reproduces mode 0 bit for bit.  The isolated
replay is address-sensitive (section 11.41), so the arms are interleaved in one
process and only ranked against each other.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_fixpipe_shape.py
"""
from __future__ import annotations

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

DEV = torch.device("npu:0")
CHUNKS = 12288
PC = api.CHUNK
M = PC // 2
MM = M * M
NC = api.ASM_NCHUNK
SLOTS = 2 * NC
GRID = (CHUNKS + NC - 1) // NC
MODES = [
    (0, "production: 4 NZ + 4 strided row-major"),
    (1, "batched: 4 NZ + 1 ndNum call"),
    (2, "8x NZ -> L1"),
    (3, "8x row-major, contiguous (stride M)"),
    (4, "8x row-major, production (stride PC)"),
    (5, "no stores (arithmetic floor)"),
]


def main() -> None:
    api._rtc("kernels/v1/k1_fixpipe_shape_probe.cpp", "kda_fixpipe_shape_probe")
    gen = torch.Generator(device="cpu").manual_seed(20260927)
    def rb(*shape):
        return (torch.randn(*shape, generator=gen) * 0.3).to(torch.bfloat16).to(DEV)
    a16 = rb(CHUNKS, PC, PC)
    xb = rb(CHUNKS, 2, M, M)
    lneg = rb(CHUNKS, M, M)
    dst3 = rb(GRID * SLOTS, M * PC)
    cur = torch_npu.npu.current_stream()
    argv = api._pack_ptrs([a16, xb, lneg, dst3])

    def launch(mode: int, src_nd: int = 4) -> None:
        api._launch("kda_fixpipe_shape_probe", GRID,
                    argv + [api._i(CHUNKS), api._i(mode), api._i(src_nd)],
                    cur.npu_stream)

    def timeit(fn, reps=5):
        xs = []
        for _ in range(reps):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.npu.synchronize()
            xs.append((time.perf_counter() - t0) * 1e3)
        return min(xs)

    print("fixpipe shape, %d chunks, grid %d, CHUNK=%d, ASM_NCHUNK=%d"
          % (CHUNKS, GRID, PC, NC), flush=True)

    launch(0)
    torch.npu.synchronize()
    ref = a16.clone()
    # only the lower-left [M, M] block of each PC x PC tile is written
    def written(t):
        return t.view(-1, PC, PC)[:, M:, :M]
    print("  srcNd sweep for mode 1 (bit-identity against mode 0):", flush=True)
    good = []
    for sn in (2, 4, 8, 16, 32):
        a16.zero_()
        torch.npu.synchronize()
        launch(1, sn)
        torch.npu.synchronize()
        nd = int((a16.view(-1, PC, PC)[:, M:, :M] != ref.view(-1, PC, PC)[:, M:, :M]).sum())
        ok = nd == 0
        if ok:
            good.append(sn)
        print("    srcNd %2d: %s (%d/%d differ)" % (sn, "IDENTICAL" if ok else "differs", nd, a16.numel()),
              flush=True)
    if not good:
        print("  no srcNd value reproduced mode 0 - batching rejected, timing the rest", flush=True)
        batch_src = None
    else:
        batch_src = good[0]
        print("  batching is legal, srcNd = %d reproduces mode 0 bit for bit" % batch_src, flush=True)

    for mode, _ in MODES:
        launch(mode)
    torch.npu.synchronize()
    best = {mode: 1e9 for mode, _ in MODES}
    for _ in range(5):
        for mode, _ in MODES:
            if mode == 1 and batch_src is None:
                continue
            best[mode] = min(best[mode], timeit(lambda m=mode: launch(m), 1))

    print()
    print("  arm                                    ms (MIN of 5)")
    for mode, name in MODES:
        if mode == 1 and batch_src is None:
            print("  mode %d %-38s (rejected)" % (mode, name))
            continue
        print("  mode %d %-38s %8.3f" % (mode, name, best[mode]), flush=True)

    print()
    print("verdict arithmetic (per call = delta / 8 stores of 2 KB)")
    print("  NZ -> L1 per-call premium       (0 - 5) / 4      %+.1f ns/call" % (1000 * (best[0] - best[5]) / 1000 / 4))
    print("  all stores vs none              mode 0 - 5        %+.3f ms" % (best[0] - best[5]))
    print("  row-stride premium              mode 4 - 3        %+.3f ms" % (best[4] - best[3]))
    print("  row-major strided per-call      (4 - 5) / 8      %+.1f ns/call"
          % (1000 * (best[4] - best[5]) / 1000 / 8))
    if batch_src is not None:
        print("  batching win (production shape) mode 0 - 1        %+.3f ms" % (best[0] - best[1]))
    print("  8x NZ -> L1 total               mode 2 - 5        %+.3f ms" % (best[2] - best[5]))


if __name__ == "__main__":
    main()
