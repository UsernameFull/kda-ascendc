"""The shipped slice policy against the legacy even split, production path.

``_launch_solve_two_level`` now sizes its slices by whole AIV waves
(docs 11.59): one wave is ``aiv_cores * nch`` chunks and the size targets
``SOLVE_SLICE_TARGET`` (20) slices, with a smaller remainder slice.  At
[1,8192,96,128] / C=64 that is 19 x 640 + 128 chunks instead of 24 even
512s, so every slice's wide grid (160 blocks) is a whole 4 waves of the 40
vector cores rather than 128 blocks = 3.2 waves; the same trimming applies to
the assemble grid (160 blocks = 8 waves of 20).

The knob is read per call (KDA_SOLVE_SLICE_CHUNKS: 0 auto, >0 fixed size,
<0 legacy), so both arms run in one process round-robin.  Prints the launch
structure of each arm (slice count and grids), the paired wall MIN, and the
bit-identity of the outputs.

  KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_solve_slice_policy.py
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
ARMS = [(0, "auto whole-wave"), (-1, "legacy even split")]
ROUNDS = int(os.environ.get("KDA_SLICE_ROUNDS", "12"))


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


def structure(q, k, v, g, beta, kw):
    seq = []
    orig = api._launch

    def spy(kernel, blocks, args, stream):
        if kernel.startswith("kda_solve"):
            seq.append((kernel, int(blocks)))
        return orig(kernel, blocks, args, stream)

    api._launch = spy
    try:
        api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    finally:
        api._launch = orig
    torch.npu.synchronize()
    wide = [b for n, b in seq if n == "kda_solve_wu_wide"]
    asm = [b for n, b in seq if n == "kda_solve_assemble"]
    cub = [b for n, b in seq if n == "kda_solve_wu_cube_kernel"]
    return wide, asm, cub


def call(q, k, v, g, beta, kw):
    torch.npu.synchronize()
    t0 = time.perf_counter()
    out, st = api.kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1e3, out, st


def main() -> None:
    q, k, v, g, beta, kw = inputs()
    call(q, k, v, g, beta, kw)  # compile + warm

    print("two-level solve slice policy at [%d,%d,%d,%d]/C=%d, "
          "%d AIV cores, wave=%d chunks" %
          (B, T, H, D, api.CHUNK, api._aiv_core_count(DEV),
           api._aiv_core_count(DEV) * api.SOLVE_WIDE_NCH), flush=True)

    ref = {}
    for mode, name in ARMS:
        os.environ["KDA_SOLVE_SLICE_CHUNKS"] = str(mode)
        wide, asm, cub = structure(q, k, v, g, beta, kw)
        print("  %-18s %2d slices: wide %s | asm %s | cube %s"
              % (name, len(wide), sorted(set(wide)), sorted(set(asm)),
                 sorted(set(cub))), flush=True)
        ref[mode] = call(q, k, v, g, beta, kw)[1:]

    wall, outs = {}, {}
    for r in range(ROUNDS):
        order = ARMS[r % len(ARMS):] + ARMS[:r % len(ARMS)]
        for mode, name in order:
            os.environ["KDA_SOLVE_SLICE_CHUNKS"] = str(mode)
            w, out, st = call(q, k, v, g, beta, kw)
            wall[mode] = min(wall.get(mode, 1e9), w)
            outs[mode] = (out, st)
    os.environ.pop("KDA_SOLVE_SLICE_CHUNKS", None)

    base = wall[ARMS[0][0]]
    for mode, name in ARMS:
        print("  %-18s wall %8.3f  delta %+7.3f" % (name, wall[mode],
                                                    wall[mode] - base), flush=True)
    out0, st0 = outs[0]
    outl, stl = outs[-1]
    print("  identity auto vs legacy: out=%d st=%d | auto vs first-call: out=%d st=%d"
          % (int((out0 != outl).sum().item()), int((st0 != stl).sum().item()),
             int((out0 != ref[0][0]).sum().item()),
             int((st0 != ref[0][1]).sum().item())), flush=True)


if __name__ == "__main__":
    main()
