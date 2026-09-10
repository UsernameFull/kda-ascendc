import sys
from pathlib import Path

import torch
import torch_npu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from kda_ascendc_v1.api import get_last_profile, kda_bt16_fwd_ascendc

torch.npu.set_device(0)
DEVICE = torch.device('npu:0')
D = 128

for b, t, h in [(1, 16, 2), (1, 64, 2), (1, 16, 32), (2, 1024, 4)]:
    torch.manual_seed(9000 + b + t + h)
    q = (torch.randn(b, t, h, D, device=DEVICE) * 0.2).to(torch.bfloat16)
    k = (torch.randn(b, t, h, D, device=DEVICE) * 0.2).to(torch.bfloat16)
    v = (torch.randn(b, t, h, D, device=DEVICE) * 0.1).to(torch.bfloat16)
    g = torch.randn(b, t, h, D, device=DEVICE) * 0.1
    beta = torch.randn(b, t, h, device=DEVICE)
    a_log = torch.linspace(-1.0, 0.2, h, device=DEVICE)
    bias = torch.randn(h, D, device=DEVICE) * 0.03
    initial_state = torch.zeros(b, h, D, D, device=DEVICE)
    common = dict(A_log=a_log, bias=bias, lower_bound=-1.0,
                  initial_state=initial_state, output_final_state=True)
    ref = kda_bt16_fwd_ascendc(q, k, v, g, beta, k2_mode='persistent', **common)
    torch.npu.synchronize()
    got = kda_bt16_fwd_ascendc(q, k, v, g, beta, k2_mode='triton_aiv', **common)
    torch.npu.synchronize()
    out_error = float((got[0] - ref[0]).abs().max().cpu())
    state_error = float((got[1] - ref[1]).abs().max().cpu())
    profile = get_last_profile()
    finite = bool(torch.isfinite(got[0]).all() and torch.isfinite(got[1]).all())
    print({'shape': [b, t, h, D], 'out_error': out_error,
           'state_error': state_error, 'finite': finite,
           'launch_counts': profile.get('launch_counts', {})}, flush=True)
