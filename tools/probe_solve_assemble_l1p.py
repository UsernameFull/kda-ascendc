"""Can the assemble's P round trip become L0C -> L1 -> L0B?

``docs/ASCENDC_V1_REFACTOR_PLAN_20260913.md`` section 11.36 priced the P round
trip at 0.137 ms of the coupling block's 0.648 ms by removing it, and section
11.37 then took the kernel's load path from 6 ND2NZ calls per chunk to 2.5 and
the stage by 0.178 ms.  This probe builds the structure that would get the
remaining 0.137 back: pass 0 fixpipes P into L1 in NZ instead of to its own GM
tile, and pass 1 builds L0B from that region.

  mode 0  shipped two-pass, P through GM (control)
  mode 1  pass 0 stores nothing and pass 1 loads nothing for B (the ceiling;
          wrong by construction)
  mode 2  the candidate, P via L0C -> L1 -> L0B

Mode 2 is only interesting if it is bit-identical to mode 0 (the NZ fractal
order is the risk - see the kernel header), so the script checks that first and
then times the three arms interleaved in one process.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=1 python3 -u tools/probe_solve_assemble_l1p.py
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
GRID = (CHUNKS + NC - 1) // NC
TILE = M * M * 2
# Bytes per chunk, in tiles: shipped = pass 0 (Lneg + Xb + P store) + pass 1
# (Xb + P load + A16 store); mode 1 drops the P store and the P load.
ARMS = [
    (0, "P through GM (control)", 6 * TILE),
    (1, "no P at all (ceiling)", 4 * TILE),
    (2, "P via L0C->L1->L0B", 4 * TILE),
]


def main() -> None:
    api._rtc("kernels/v1/k1_solve_assemble_l1p_probe.cpp", "kda_solve_assemble_l1p_probe")
    gen = torch.Generator(device="cpu").manual_seed(20260923)
    def rb(*shape):
        return (torch.randn(*shape, generator=gen) * 0.3).to(torch.bfloat16).to(DEV)
    a16 = rb(CHUNKS, PC, PC)
    xb = rb(CHUNKS, 2, M, M)
    lneg = rb(CHUNKS, M, M)
    p = rb(CHUNKS, M, M)
    cur = torch_npu.npu.current_stream()
    argv = api._pack_ptrs([a16, xb, lneg, p])

    def launch(mode: int) -> None:
        api._launch("kda_solve_assemble_l1p_probe", GRID,
                    argv + [api._i(CHUNKS), api._i(mode)], cur.npu_stream)

    def timeit(fn, reps=5):
        xs = []
        for _ in range(reps):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.npu.synchronize()
            xs.append((time.perf_counter() - t0) * 1e3)
        return min(xs)

    print("assemble P round trip, %d chunks, grid %d, CHUNK=%d, ASM_NCHUNK=%d"
          % (CHUNKS, GRID, PC, NC), flush=True)

    launch(0)
    torch.npu.synchronize()
    ref_a16, ref_p = a16.clone(), p.clone()
    launch(2)
    torch.npu.synchronize()
    same_a = torch.equal(a16, ref_a16)
    same_p = torch.equal(p, ref_p)
    nd = int((a16 != ref_a16).sum())
    print("  mode 2 vs mode 0: A16 %s (%d/%d differ), P tile %s"
          % ("IDENTICAL" if same_a else "DIFFERS", nd, a16.numel(),
             "left alone" if same_p else "written"), flush=True)

    for mode, _, _ in ARMS:
        launch(mode)
    torch.npu.synchronize()
    best = {mode: 1e9 for mode, _, _ in ARMS}
    for _ in range(5):
        for mode, _, _ in ARMS:
            best[mode] = min(best[mode], timeit(lambda m=mode: launch(m), 1))

    print()
    print("  arm                       bytes/call   ms (MIN of 5)   GB/s")
    for mode, name, per_chunk in ARMS:
        nb = per_chunk * CHUNKS / 1e6
        print("  mode %d %-20s %7.1f MB   %8.3f      %6.1f"
              % (mode, name, nb, best[mode], nb / best[mode]), flush=True)
    print()
    print("verdict")
    print("  the round trip's ceiling (0 - 1): %+.3f ms" % (best[0] - best[1]))
    print("  the candidate          (0 - 2): %+.3f ms  (%.1f%% of the control)"
          % (best[0] - best[2], 100.0 * best[2] / best[0]))
    print("  the candidate vs that ceiling:  %+.3f ms" % (best[2] - best[1]))


if __name__ == "__main__":
    main()
