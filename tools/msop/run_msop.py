"""Drive one KDA kernel for on-board msopprof collection.

msopprof's injection library interposes `aclrtLaunchKernel*` and
`rtKernelLaunch` but not `aclrtLaunchKernelWithArgsArray`, which is what the
production launcher uses - so profiling the production path shows nothing.
This driver compiles the kernel through `tools/msop/msop_shim.cpp` (same aclrtc
options, but launched through the interposed call) and runs it, so msopprof
sees the kernel and can collect on-board metrics for it.

  KDA_CHUNK=64 KDA_MSOPP_KERNEL=kda_solve_assemble \\
    msopprof --output=<dir> python3 tools/msop/run_msop.py

Shapes match every other probe here ([1, 8192, 96, 128], C=64) so the numbers
are comparable with the docs.
"""
from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import torch
import torch_npu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "build/msop_shim"))
import kda_ascendc_v1.api as api  # noqa: E402
import msop_shim  # noqa: E402

DEV = torch.device("npu:0")
SRC = "kernels/v1/k1_solve_assemble.cpp"


def main() -> None:
    name = os.environ.get("KDA_MSOPP_KERNEL", "kda_solve_assemble")
    iters = int(os.environ.get("KDA_MSOPP_ITERS", "3"))
    pc, m = api.CHUNK, api.CHUNK // 2
    chunks = 1 * 96 * (8192 // pc)
    a16 = torch.zeros(chunks, pc, pc, dtype=torch.bfloat16, device=DEV)
    xb = (torch.randn(chunks, 2, m, m, device=DEV) * 0.1).to(torch.bfloat16)
    lneg = (torch.randn(chunks, m, m, device=DEV) * 0.1).to(torch.bfloat16)
    torch.npu.synchronize()

    src = (ROOT / SRC).read_text(encoding="utf-8-sig")
    handle = msop_shim.rtc_handle(api._defines() + src, name)
    nc = api.ASM_NCHUNK
    grid = (chunks + nc - 1) // nc
    # aclrtLaunchKernel takes the arguments packed into one contiguous buffer in
    # declaration order: 4 pointers, then the two int32 parameters.
    # mode 2 keeps P on chip, so the production caller passes a null there
    blob = b"".join(struct.pack("<Q", t.data_ptr()) for t in (a16, xb, lneg))
    blob += struct.pack("<Q", 0)
    blob += struct.pack("<ii", chunks, 2)
    stream = torch_npu.npu.current_stream().npu_stream
    for _ in range(iters):
        msop_shim.launch(handle, grid, blob, stream)
    torch.npu.synchronize()
    print("launched %s x%d (grid %d, %d chunks)" % (name, iters, grid, chunks), flush=True)


if __name__ == "__main__":
    main()
