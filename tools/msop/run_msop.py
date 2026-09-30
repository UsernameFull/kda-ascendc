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

# kernel -> (source, default mode); the assemble blobs are the four pointers
# [a16, xb, lneg, null] followed by (C, mode), so a probe arm is picked with
# KDA_MSOPP_MODE instead of a second driver.  The wide solve is the AIV-side
# kernel and takes six pointers [L, eye, a32, a16, xb, lneg] followed by
# (C, a16Mode, debugStores) - its "mode" is the A16-store ablation
# (KDA_SOLVE_A16_MODE, 0 = production).
GEOMETRY = {
    "kda_solve_assemble": ("kernels/v1/k1_solve_assemble.cpp", 2),
    # The SB = 4 coupling kernel: three pointers [a16, xb, lneg] then
    # (C, storeMode, ab, level) - the last two are the debug knobs and are
    # pinned to their production values here (no ablation, all six levels).
    "kda_solve_assemble4": ("kernels/v1/k1_solve_assemble4.cpp", 0),
    "kda_solve_assemble_pipe_probe":
        ("kernels/v1/k1_solve_assemble_pipe_probe.cpp", 0),
    "kda_solve_wu_wide": ("kernels/v1/k1_solve_wu_wide.cpp", 0),
}


def main() -> None:
    name = os.environ.get("KDA_MSOPP_KERNEL", "kda_solve_assemble")
    iters = int(os.environ.get("KDA_MSOPP_ITERS", "3"))
    src_name, default_mode = GEOMETRY[name]
    mode = int(os.environ.get("KDA_MSOPP_MODE", str(default_mode)))
    pc, m = api.CHUNK, api.CHUNK // 2
    chunks = 1 * 96 * (8192 // pc)
    a16 = torch.zeros(chunks, pc, pc, dtype=torch.bfloat16, device=DEV)
    xb = (torch.randn(chunks, 2, m, m, device=DEV) * 0.1).to(torch.bfloat16)
    lneg = (torch.randn(chunks, m, m, device=DEV) * 0.1).to(torch.bfloat16)
    torch.npu.synchronize()

    src = (ROOT / src_name).read_text(encoding="utf-8-sig")
    handle = msop_shim.rtc_handle(api._defines() + src, name)
    if name == "kda_solve_wu_wide":
        # Six pointers + (C, a16Mode, debugStores), grid over the wide kernel's
        # own block size NCH = NCHUNK / SUBB chunks.  L is the fp32 Gram tile
        # (zeros is well-formed: the recursion then lands exactly on the
        # identity), a32 is only touched under debugStores != 0.
        subb = api.SOLVE_WIDE_SUBB
        sub = pc // subb
        L = torch.zeros(chunks, pc, pc, dtype=torch.float32, device=DEV)
        eye = torch.eye(sub, dtype=torch.float32, device=DEV)
        a32 = torch.zeros(sub, sub, dtype=torch.float32, device=DEV)
        xb = torch.zeros(chunks, subb, sub, sub, dtype=torch.bfloat16,
                         device=DEV)
        blob = b"".join(struct.pack("<Q", t.data_ptr())
                        for t in (L, eye, a32, a16, xb, lneg))
        blob += struct.pack("<iii", chunks, mode, 0)
        grid = (chunks + (api.SOLVE_WIDE_NCHUNK // subb) - 1) // \
            (api.SOLVE_WIDE_NCHUNK // subb)
    elif name == "kda_solve_assemble4":
        # Three pointers then (C, storeMode, ab, level); xb/lneg are sized for
        # the SB = 4 export (four diagonal tiles and the six-block bundle) and
        # the pointers only have to be well-formed, since msopprof reads the
        # schedule, not the values.
        subb = api.SOLVE_WIDE_SUBB
        sub = pc // subb
        xb = torch.zeros(chunks, subb, sub, sub, dtype=torch.bfloat16, device=DEV)
        lneg = torch.zeros(chunks, 6 * sub * sub, dtype=torch.bfloat16, device=DEV)
        nc = api.ASM_NCHUNK
        grid = (chunks + nc - 1) // nc
        blob = b"".join(struct.pack("<Q", t.data_ptr()) for t in (a16, xb, lneg))
        blob += struct.pack("<iiii", chunks, mode, 0, 6)
    else:
        # aclrtLaunchKernel takes the arguments packed into one contiguous
        # buffer in declaration order: 4 pointers, then the two int32
        # parameters.  mode 2 keeps P on chip and the probe never reads P, so
        # the fourth pointer is null in both geometries
        nc = api.ASM_NCHUNK
        grid = (chunks + nc - 1) // nc
        blob = b"".join(struct.pack("<Q", t.data_ptr()) for t in (a16, xb, lneg))
        blob += struct.pack("<Q", 0)
        blob += struct.pack("<ii", chunks, mode)
    stream = torch_npu.npu.current_stream().npu_stream
    for _ in range(iters):
        msop_shim.launch(handle, grid, blob, stream)
    torch.npu.synchronize()
    print("launched %s x%d (grid %d, %d chunks, mode %d)"
          % (name, iters, grid, chunks, mode), flush=True)


if __name__ == "__main__":
    main()
