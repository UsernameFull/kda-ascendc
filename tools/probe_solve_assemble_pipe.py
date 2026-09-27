"""Does the assemble's block wall have to be the sum of its pipes?

The on-board profile (docs section 11.42) reads the coupling-block kernel's
block wall as the sum of its pipe busy times: fixpipe 1.812 + mte2 1.201 +
scalar 0.610 + mte1 0.162 + cube 0.098 = 3.884 us against a 3.652 us wall,
i.e. no two pipes are ever busy at the same time.  The divide is the per-unit
drain chain: LoadData -> SetFlag/WaitFlag<MTE1_M> -> Mmad -> SetFlag/WaitFlag
<M_FIX> -> Fixpipe, eight times per block, plus the two-pass barrier.

The units are independent (one L0/L0C slot each), so the drains are the only
thing serialising them.  This probe is the production mode-2 structure with
the drains removed one layer at a time; all arms compute and store the same
thing, so A16 must be bit-identical in every arm.

  mode 0  control: production chain (per-unit drains, pass barrier, pass 1's
          la issued at the top of pass 1)
  mode 1  hoist: pass 1's la issued before pass 0's loads
  mode 2  phases: hoist + per pass: all fills, one MTE1_M wait, all Mmads,
          one M_FIX wait, all Fixpipes
  mode 3  interleave: hoist + ping-pong M_FIX ids so Fixpipe ch overlaps
          Mmad ch+1 (<= 1 outstanding set per id)
  mode 4  phases without the hoist (separates the drain change from it)

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_solve_assemble_pipe.py
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
MODES = [
    (0, "control (production chain)"),
    (1, "hoist pass-1 la"),
    (2, "phases + hoist"),
    (3, "interleave (ping-pong M_FIX)"),
    (4, "phases, no hoist"),
    (5, "depth-1 load pipeline (per-chunk waits)"),
]


def main() -> None:
    api._rtc("kernels/v1/k1_solve_assemble_pipe_probe.cpp",
             "kda_solve_assemble_pipe_probe")
    gen = torch.Generator(device="cpu").manual_seed(20260927)
    def rb(*shape):
        return (torch.randn(*shape, generator=gen) * 0.3).to(torch.bfloat16).to(DEV)
    a16 = rb(CHUNKS, PC, PC)
    xb = rb(CHUNKS, 2, M, M)
    lneg = rb(CHUNKS, M, M)
    p = rb(CHUNKS, M, M)
    cur = torch_npu.npu.current_stream()
    argv = api._pack_ptrs([a16, xb, lneg, p])

    def launch(mode: int) -> None:
        api._launch("kda_solve_assemble_pipe_probe", GRID,
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

    print("assemble pipe overlap, %d chunks, grid %d, CHUNK=%d, ASM_NCHUNK=%d"
          % (CHUNKS, GRID, PC, NC), flush=True)
    launch(0)
    torch.npu.synchronize()
    ref = a16.clone()
    # only the lower-left [M, M] block of each PC x PC tile is written
    def written(t):
        return t.view(-1, PC, PC)[:, M:, :M]
    refp = p.clone()
    for mode, name in MODES[1:]:
        a16.zero_()
        torch.npu.synchronize()
        launch(mode)
        torch.npu.synchronize()
        nd = int((a16.view(-1, PC, PC)[:, M:, :M] != ref.view(-1, PC, PC)[:, M:, :M]).sum())
        print("  mode %d %-30s A16 %s (%d/%d differ), P %s"
              % (mode, name, "IDENTICAL" if nd == 0 else "DIFFERS", nd,
                 a16.numel(), "left alone" if torch.equal(p, refp) else "written"),
              flush=True)

    for mode, _ in MODES:
        launch(mode)
    torch.npu.synchronize()
    best = {mode: 1e9 for mode, _ in MODES}
    for _ in range(5):
        for mode, _ in MODES:
            best[mode] = min(best[mode], timeit(lambda m=mode: launch(m), 1))

    print()
    print("  arm                                  ms (MIN of 5)   vs control")
    for mode, name in MODES:
        print("  mode %d %-30s %8.3f        %+.3f"
              % (mode, name, best[mode], best[mode] - best[0]), flush=True)
    print()
    print("  hoist alone            (0 - 1): %+.3f ms" % (best[0] - best[1]))
    print("  phases + hoist         (0 - 2): %+.3f ms" % (best[0] - best[2]))
    print("  interleave + hoist     (0 - 3): %+.3f ms" % (best[0] - best[3]))
    print("  phases alone           (0 - 4): %+.3f ms" % (best[0] - best[4]))
    print("  depth-1 pipeline       (0 - 5): %+.3f ms" % (best[0] - best[5]))


if __name__ == "__main__":
    main()
