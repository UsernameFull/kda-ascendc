import struct
import sys
from pathlib import Path

import torch
import torch_npu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from kda_ascendc_v1.api import launch_argsarray_engine, rtc_compile

torch.npu.set_device(0)
rtc_compile((ROOT / 'kernels/v1/k2_triton_cube_aic_probe.cpp').read_text(),
            'kda_k2_triton_cube_aic_probe', '')
device = torch.device('npu:0')
blocks, m, n, k = 2, 16, 64, 128
a = (torch.randn(blocks, m, k, device=device) * 0.2).to(torch.bfloat16)
b = (torch.randn(blocks, n, k, device=device) * 0.2).to(torch.bfloat16)
out = torch.empty(blocks, m, n, dtype=torch.float32, device=device)
ptrs = [struct.pack('<Q', int(x.data_ptr())) for x in (a, b, out)]
args = ptrs + [struct.pack('<i', blocks)]
launch_argsarray_engine('kda_k2_triton_cube_aic_probe', blocks,
                        torch.npu.current_stream().npu_stream, args, 0)
torch.npu.synchronize()
ref = torch.matmul(a.float(), b.float().transpose(-1, -2))
error = float((out - 3.0 * ref).abs().max().cpu())
print({'max_abs': error, 'finite': bool(torch.isfinite(out).all())}, flush=True)
assert error < 0.2
