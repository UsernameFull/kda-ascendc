"""The wide kernel's chunks-per-block knob (KDA_SOLVE_WIDE_NCHUNK).

Section 11.51 put the AIV half of the solve on board: 68% of the wide block's
wall is `vec` - the row recursion - and its instruction count is one `Adds` +
one `Brcb` per row plus one `MulAddDst` per (row i, row j < i) pair *per tile*,
while the repeats of every one of those instructions walk the tile's
(chunk, sub-block) instances.  So the per-chunk instruction count is

    M(M + 3) / 2 / NCH          (NCH = NCHUNK / SUBB chunks per block)

i.e. it falls with the chunks a block covers, and NCHUNK is a compile-time
constant with an env override.  Going 8 -> 12 takes 140 -> 93 instructions per
chunk; 8 -> 16 takes it to 70.  Unlike SUBB = 4 (which moves the coupling to
the Cube and needs a new assemble), this knob leaves the coupling shape alone:
still one off-diagonal block per chunk, ASM_NCHUNK = 4, same parents tiles.
The constraint is UB: the tile is SB*NCH instances deep, so every live tile
grows linearly in NCHUNK (lraw/af/ab/cexp/L21 sum to ~134 KB at NCHUNK = 8 on
a 192 KB UB).

This probe prices the knob in isolation, on the production shape
([1, 8192, 96, 128], 12288 chunks), by compiling the wide kernel at whatever
KDA_SOLVE_WIDE_NCHUNK says and replaying exactly its own launch (six pointers
+ (C, a16Mode, debugStores), grid = ceil(C / NCH)) MIN of 5.

The outputs are chunk-local, so a different NCHUNK must produce bit-identical
A16 / Xb / Lneg: the probe digests the first 64 chunks and the arms are
compared against each other by re-running it with another NCHUNK (dump files
land in /tmp/wide_nchunk_<n>_<part>.pt).

  KDA_CHUNK=64 KDA_SOLVE_WIDE_NCHUNK=12 ASCEND_RT_VISIBLE_DEVICES=3 \
    python3 -u tools/probe_solve_wide_nchunk.py
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
    m = pc // sb
    nchunk = api.SOLVE_WIDE_NCHUNK
    nch = nchunk // sb
    chunks = B * H * (T // pc)
    assert chunks % nch == 0, (chunks, nch)
    print("geometry: CHUNK %d SUBB %d M %d NCHUNK %d NCH %d chunks %d grid %d"
          % (pc, sb, m, nchunk, nch, chunks, chunks // nch), flush=True)

    torch.manual_seed(1312)
    # A strictly-lower-triangular fp32 L per chunk: exactly what the recursion
    # consumes (diagonal implied by the identity), value-dependent nowhere in
    # the schedule, and chunk-local so NCHUNK cannot change a single output.
    L = (torch.randn(chunks, pc, pc, device=DEV) * 0.1).tril(-1).contiguous()
    eye = torch.eye(m, dtype=torch.float32, device=DEV)
    a32 = torch.zeros(m, m, dtype=torch.float32, device=DEV)
    a16 = torch.zeros(chunks, pc, pc, dtype=torch.bfloat16, device=DEV)
    xb = torch.zeros(chunks, sb, m, m, dtype=torch.bfloat16, device=DEV)
    ce = (6 * m * m) if sb == 4 else m * m
    lneg = torch.zeros(chunks, ce, dtype=torch.bfloat16, device=DEV)
    torch.npu.synchronize()

    api._rtc("kernels/v1/k1_solve_wu_wide.cpp", "kda_solve_wu_wide")
    args = api._pack_ptrs([L, eye, a32, a16, xb, lneg]) + \
        [api._i(chunks), api._i(0), api._i(0)]
    grid = chunks // nch
    stream = torch_npu.npu.current_stream()

    def run():
        api.launch_argsarray_engine("kda_solve_wu_wide", grid,
                                    stream.npu_stream, args, 0)

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
    print("wide replay: MIN %.3f ms  (all: %s)  aggregate %.1f ns/chunk  "
          "(grid %d)" % (xs[0], " ".join("%.3f" % x for x in xs),
                         xs[0] * 1e6 / chunks, grid), flush=True)

    torch.save({"a16": a16[:64].cpu(), "xb": xb[:64].cpu(),
                "lneg": lneg[:64].cpu()},
               "/tmp/wide_sb%d_nchunk%d.pt" % (sb, nchunk))

    # Host-side gate at any SUBB: the diagonal sub-block of a block lower
    # triangular inverse is the inverse of that diagonal block, so the kernel's
    # A16 must equal the fp64 reference on every (s, s) block - and the strict
    # upper triangle must be exactly the blank.  bf16 output vs fp64 reference,
    # so the tolerance is one bf16 ulp-class number rather than bit equality.
    ref = torch.linalg.inv(torch.eye(pc, dtype=torch.float64)
                           + L[:64].cpu().double())
    a16h = a16[:64].cpu().double()
    worst = 0.0
    for s in range(sb):
        got = a16h[:, s * m:(s + 1) * m, s * m:(s + 1) * m]
        want = ref[:, s * m:(s + 1) * m, s * m:(s + 1) * m]
        worst = max(worst, float((got - want).abs().max()))
    upper = float(a16h.triu(diagonal=1).abs().max())
    print("gate: diag blocks vs fp64 inv max|d| %.3e   strict-upper max|v| %.3e"
          % (worst, float(upper)), flush=True)
    if sb == 4:
        # The export bundle must be exactly the negated 16-blocks, in bf16:
        # [ (1,0) | (3,2) | rows 32..63 x cols 0..31 ] per chunk.
        Lc = L[:64]
        want = torch.cat([
            -Lc[:, m:2 * m, 0:m].reshape(64, -1),
            -Lc[:, 3 * m:4 * m, 2 * m:3 * m].reshape(64, -1),
            -Lc[:, 2 * m:4 * m, 0:2 * m].reshape(64, -1)], dim=1)
        want = want.to(torch.bfloat16)
        got = lneg[:64]
        print("gate: Lneg bundle bit-equal %s (%d elements)"
              % (bool(torch.equal(got, want)), got.numel()), flush=True)
    print("digest: a16 sum %.6f abs %.6f  xb sum %.6f abs %.6f  lneg sum %.6f abs %.6f"
          % (float(a16[:64].float().sum()), float(a16[:64].float().abs().sum()),
             float(xb[:64].float().sum()), float(xb[:64].float().abs().sum()),
             float(lneg[:64].float().sum()), float(lneg[:64].float().abs().sum())),
          flush=True)


if __name__ == "__main__":
    main()
