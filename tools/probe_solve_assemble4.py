"""Isolated pricing of the SB = 4 coupling kernel (kda_solve_assemble4).

Runs the real wide kernel first (the production SB = 4 geometry) so the
assembly under test sees the exact operands it gets in a call - the diagonal
tiles and the blank in A16, the four Xb tiles and the six-block Lneg bundle -
then gates the six coupling blocks against an fp64 inverse of the same L and
times the coupling launch alone (MIN of 5, whole grid).

The gate is a tolerance, not bit equality: the SB = 4 schedule rounds its
intermediates (X10, X20, X21, Ea, Gr, Gl) through bf16 on the fixpipe path,
which the 32-level two-pass form does not do for the same values.  What has to
hold is the block math: every one of the six strictly-lower 16-blocks equals
the corresponding block of inv(I + L), and the strict upper triangle stays
zero.

  KDA_CHUNK=64 KDA_SOLVE_WIDE_SUBB=4 KDA_SOLVE_WIDE_NCHUNK=32 \
    ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_solve_assemble4.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import torch_npu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import kda_ascendc_v1.api as api  # noqa: E402

DEV = torch.device("npu:0")
B, T, H, D = 1, 8192, 96, 128


def main() -> None:
    pc = api.CHUNK
    sb = api.SOLVE_WIDE_SUBB
    assert sb == 4 and pc == 64, (sb, pc)
    m = pc // sb
    mm = m * m
    ce = 6 * mm
    nchunk = api.SOLVE_WIDE_NCHUNK
    nch = nchunk // sb
    nc = api.ASM_NCHUNK
    chunks = B * H * (T // pc)
    print("geometry: CHUNK %d SUBB %d M %d NCHUNK %d NCH %d ASM_NC %d chunks %d "
          "grids wide %d / asm4 %d" % (pc, sb, m, nchunk, nch, nc, chunks,
                                       chunks // nch, chunks // nc), flush=True)

    torch.manual_seed(1312)
    L = (torch.randn(chunks, pc, pc, device=DEV) * 0.1).tril(-1).contiguous()
    eye = torch.eye(m, dtype=torch.float32, device=DEV)
    a32 = torch.zeros(pc, pc, dtype=torch.float32, device=DEV)
    a16 = torch.zeros(chunks, pc, pc, dtype=torch.bfloat16, device=DEV)
    xb = torch.zeros(chunks, sb, m, m, dtype=torch.bfloat16, device=DEV)
    lneg = torch.zeros(chunks, ce, dtype=torch.bfloat16, device=DEV)
    torch.npu.synchronize()

    api._rtc("kernels/v1/k1_solve_wu_wide.cpp", "kda_solve_wu_wide")
    api._rtc("kernels/v1/k1_solve_assemble4.cpp", "kda_solve_assemble4")
    stream = torch_npu.npu.current_stream()

    wargs = api._pack_ptrs([L, eye, a32, a16, xb, lneg]) + \
        [api._i(chunks), api._i(0), api._i(0)]
    api.launch_argsarray_engine("kda_solve_wu_wide", chunks // nch,
                                stream.npu_stream, wargs, 0)
    torch.npu.synchronize()

    # The kernel's two trailing knobs are debug-only (an instruction-group
    # ablation mask and a level bisect) and default to 0 / 6; they are passed
    # explicitly because the launch ABI is positional.
    aargs = api._pack_ptrs([a16, xb, lneg]) + \
        [api._i(chunks), api._i(api.asm4_store_mode()), api._i(api.asm4_ablate()),
         api._i(api.asm4_level())]
    grid = chunks // nc

    def run():
        api.launch_argsarray_engine("kda_solve_assemble4", grid,
                                    stream.npu_stream, aargs, 0)

    for _ in range(3):
        run()
    torch.npu.synchronize()
    xs = []
    for _ in range(5):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        run()
        torch.npu.synchronize()
        xs.append((time.perf_counter() - t0) * 1e3)
    xs.sort()
    print("assemble4 replay: MIN %.3f ms  (all: %s)  per chunk %.1f ns  per block %.3f us"
          % (xs[0], " ".join("%.3f" % x for x in xs), xs[0] * 1e6 / chunks,
             xs[0] * 1e3 / grid), flush=True)

    # Host gate at fp64: the full inverse, block by block.
    h = a16[:64].cpu().float()
    ref = torch.linalg.inv(torch.eye(pc, dtype=torch.float64)
                           + L[:64].cpu().double()).float()
    worst, where = 0.0, None
    for s2 in range(sb):
        for s1 in range(s2 + 1):
            got = h[:, s2 * m:(s2 + 1) * m, s1 * m:(s1 + 1) * m]
            want = ref[:, s2 * m:(s2 + 1) * m, s1 * m:(s1 + 1) * m]
            d = float((got - want).abs().max())
            if d > worst:
                worst, where = d, (s2, s1)
    upper = float(h.triu(diagonal=1).abs().max())
    print("gate: all 16-blocks vs fp64 max|d| %.3e (at %s)  strict-upper max|v| %.3e"
          % (worst, where, upper), flush=True)
    print("digest: a16 sum %.6f abs %.6f" % (float(h.sum()), float(h.abs().sum())),
          flush=True)
    torch.save({"a16": h}, "/tmp/asm4_a16_sb%d_nchunk%d.pt" % (sb, nchunk))


if __name__ == "__main__":
    main()
