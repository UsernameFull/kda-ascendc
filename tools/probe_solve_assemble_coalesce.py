"""Same bytes, fewer calls: does the assemble's 297 GB/s come from call count?

``docs/ASCENDC_V1_REFACTOR_PLAN_20260913.md`` section 11.36 measured the
assemble's GM traffic alone at 0.339 ms for 100.7 MB (297 GB/s) against the cube
solve's 839 GB/s cold-path marginal rate, with the shipped pattern issuing six
ND2NZ calls per chunk (one 2 KB whole-tile A load plus two 1 KB B bands per
pass, ave 1.37 KB per call).  This probe holds the bytes fixed and varies only
the number of calls, using ``kernels/v1/k1_solve_assemble_coalesce_probe.cpp``:

  mode 0  shipped pattern                          6 calls/chunk   73728 total
  mode 1  the two B bands merged per chunk         4 calls/chunk   49152
  mode 2  the A load batched over the block        2.5 calls/chunk 15360
  mode 3  both batched from a packed source          1 call/chunk  12288
  mode 4  the whole block-pass in one call         0.5 calls/chunk  6144
  mode 5  the full structure (TBuf, one flag per pass), shipped pattern
  mode 6  mode 5's structure with mode 2's loads
  mode 7  the shipped kernel verbatim: per-chunk L1 queue, shipped calls
  mode 8  the shipped queue structure with mode 2's loads (the candidate)

Modes 0-4 are load-only measurement arms.  Mode 5/6 are the real kernel
structure, and the script checks that mode 6 is bit-identical to mode 5 before
timing anything - a batched ND2NZ that lands different bytes in L1 is not a
candidate no matter how fast it is.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=1 python3 -u tools/probe_solve_assemble_coalesce.py
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

D, DEV = 128, torch.device("npu:0")
B, T, H = 1, 8192, 96
CHUNKS = 12288
PC = api.CHUNK
M = PC // 2
MM = M * M
NC = api.ASM_NCHUNK
GRID = (CHUNKS + NC - 1) // NC
# Per chunk-pass: one [M, M] A tile (MM elements) + two [16, M] B bands
# (2 * BANDE elements).  73728 = 6 * 12288 calls, so every arm moves 100.7 MB.
BANDE = (M // 16) * 256
ARMS = [
    (0, "shipped pattern", 6.0, None),
    (1, "B bands merged", 4.0, None),
    (2, "A batched per block", 2.5, None),
    (3, "both batched (packed)", 1.0, None),
    (4, "one call per block-pass", 0.5, None),
    (5, "TBuf structure, shipped", 6.0, "structure"),
    (6, "TBuf structure, batched", 2.5, "structure"),
    (7, "queue structure, shipped", 6.0, "structure"),
    (8, "queue structure, batched", 2.5, "structure"),
]
LOADS_ONLY = [0, 1, 2, 3, 4]
MB = 100.7


def main() -> None:
    api._rtc("kernels/v1/k1_solve_assemble_coalesce_probe.cpp",
             "kda_solve_assemble_coalesce_probe")
    gen = torch.Generator(device="cpu").manual_seed(20260923)
    def rb(*shape):
        return (torch.randn(*shape, generator=gen) * 0.2).to(torch.bfloat16).to(DEV)
    a16 = rb(CHUNKS, PC, PC)
    xb = rb(CHUNKS, 2, M, M)
    lneg = rb(CHUNKS, M, M)
    p = rb(CHUNKS, M, M)
    pack = torch.zeros(GRID * 2 * 2 * NC * MM, dtype=torch.bfloat16, device=DEV)
    cur = torch_npu.npu.current_stream()
    cur_h = cur.npu_stream
    argv = api._pack_ptrs([a16, xb, lneg, p, pack])

    def launch(mode: int) -> None:
        api._launch("kda_solve_assemble_coalesce_probe", GRID,
                    argv + [api._i(CHUNKS), api._i(mode)], cur_h)

    def timeit(fn, reps=5):
        xs = []
        for _ in range(reps):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.npu.synchronize()
            xs.append((time.perf_counter() - t0) * 1e3)
        return min(xs)

    print("assemble coalescing, %d chunks, grid %d, CHUNK=%d, ASM_NCHUNK=%d"
          % (CHUNKS, GRID, PC, NC))

    # ---- the candidates have to be the same arithmetic, not just fewer calls -
    def outputs(mode):
        launch(mode)
        torch.npu.synchronize()
        return p.clone(), a16.clone()

    r7p, r7a = outputs(7)
    r8p, r8a = outputs(8)
    print("  bit-identity mode 8 vs mode 7 (queue structure): P %s, A16 %s"
          % ("IDENTICAL" if torch.equal(r8p, r7p) else "DIFFERS",
             "IDENTICAL" if torch.equal(r8a, r7a) else "DIFFERS"), flush=True)
    r5p, r5a = outputs(5)
    r6p, r6a = outputs(6)
    print("  bit-identity mode 6 vs mode 5 (TBuf structure):  P %s, A16 %s"
          % ("IDENTICAL" if torch.equal(r6p, r5p) else "DIFFERS",
             "IDENTICAL" if torch.equal(r6a, r5a) else "DIFFERS"), flush=True)
    print("  cross-structure (mode 7 vs mode 5):              P %s, A16 %s"
          % ("IDENTICAL" if torch.equal(r7p, r5p) else "DIFFERS",
             "IDENTICAL" if torch.equal(r7a, r5a) else "DIFFERS"), flush=True)

    for mode, _, _, _ in ARMS:
        launch(mode)
    torch.npu.synchronize()
    best = {mode: 1e9 for mode, _, _, _ in ARMS}
    for _ in range(5):
        for mode, _, _, _ in ARMS:
            best[mode] = min(best[mode], timeit(lambda m=mode: launch(m), 1))

    print()
    print("  arm                        calls   calls/chunk   ms (MIN of 5)   GB/s")
    for mode, name, cpc, _ in ARMS:
        calls = int(round(cpc * CHUNKS))
        print("  mode %d %-20s %6d  %8.2f     %8.3f      %6.1f"
              % (mode, name, calls, cpc, best[mode], MB / best[mode]), flush=True)

    print()
    print("verdict arithmetic (loads-only arms, identical 100.7 MB)")
    for mode, name, cpc, _ in ARMS:
        if mode in LOADS_ONLY:
            print("  mode %d %-22s %5.1f calls/chunk  %7.3f ms  vs mode 0 %+.3f  %6.1f GB/s"
                  % (mode, name, cpc, best[mode], best[mode] - best[0], MB / best[mode]))
    print("  structure, TBuf  (mode 5 -> 6): %.3f -> %.3f ms  (%+.3f, %.1f%%)"
          % (best[5], best[6], best[6] - best[5],
             100.0 * (best[6] - best[5]) / best[5]))
    print("  structure, queue (mode 7 -> 8): %.3f -> %.3f ms  (%+.3f, %.1f%%)"
          % (best[7], best[8], best[8] - best[7],
             100.0 * (best[8] - best[7]) / best[7]))
    print("  structure cost of TBuf vs queue at the shipped calls (5 - 7): %+.3f ms"
          % (best[5] - best[7]))
    print("  best candidate (mode 8) vs the shipped queue structure (mode 7): %+.3f ms"
          % (best[8] - best[7]))
    print("  per-call price from mode 0: %.2f ns/call (%.4f ms / %d calls)"
          % (1e6 * best[0] / (6 * CHUNKS), best[0], 6 * CHUNKS))


if __name__ == "__main__":
    main()
