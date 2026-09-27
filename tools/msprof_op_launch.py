"""Launch one named RTC kernel directly, for `msprof op`.

Profiling the full pipeline means compiling all 25 RTC kernels and then
filtering thousands of torch_npu launches; this script compiles exactly the
kernel named by ``KDA_MSOPP_KERNEL`` and launches it ``KDA_MSOPP_ITERS`` times
on dummy tensors, so the capture is the kernel and nothing else.

  KDA_CHUNK=64 KDA_MSOPP_KERNEL=kda_solve_assemble KDA_MSOPP_ITERS=5 \\
    msprof op --application="python3 tools/msprof_op_launch.py" \\
              --aic-metrics=PipeUtilization --kernel-name=kda_solve_assemble \\
              --launch-count=1 --output=<dir>

The dummy operands match the production shapes at [1,8192,96,128]/C=64; the
kernel's arithmetic is meaningless here on purpose (msprof replays the kernel
with the same arguments, so the data only has to be well-formed).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch_npu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import kda_ascendc_v1.api as api  # noqa: E402

DEV = torch.device("npu:0")

# kernel -> (source, extra args beyond the shared [a16, xb, lneg] trio)
GEOMETRY = {
    "kda_solve_assemble": ("kernels/v1/k1_solve_assemble.cpp", 4),
}


def main() -> None:
    name = os.environ.get("KDA_MSOPP_KERNEL", "kda_solve_assemble")
    iters = int(os.environ.get("KDA_MSOPP_ITERS", "5"))
    src, nptr = GEOMETRY[name]
    api._rtc(src, name)

    pc, m = api.CHUNK, api.CHUNK // 2
    chunks = 1 * 96 * (8192 // pc)
    a16 = torch.zeros(chunks, pc, pc, dtype=torch.bfloat16, device=DEV)
    xb = (torch.randn(chunks, 2, m, m, device=DEV) * 0.1).to(torch.bfloat16)
    lneg = (torch.randn(chunks, m, m, device=DEV) * 0.1).to(torch.bfloat16)
    pmid = torch.zeros(chunks, m, m, dtype=torch.bfloat16, device=DEV)
    nc = api.ASM_NCHUNK
    args = api._pack_ptrs([a16, xb, lneg, None]) + [api._i(chunks), api._i(2)]
    grid = (chunks + nc - 1) // nc
    stream = torch_npu.npu.current_stream().npu_stream
    assert nptr == 4, "kernel signature changed"
    for _ in range(iters):
        api._launch(name, grid, args, stream)
    torch.npu.synchronize()
    print("launched %s x%d (grid %d, %d chunks)" % (name, iters, grid, chunks),
          flush=True)


if __name__ == "__main__":
    main()
