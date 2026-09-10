import json
import os
import sys
import time
from pathlib import Path

import torch
import torch_npu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from kda_ascendc_v1.api import kda_bt16_fwd_ascendc

torch.npu.set_device(0)
device = torch.device('npu:0')
b = int(os.environ.get('CUBE_B', '2'))
t = int(os.environ.get('CUBE_T', '1024'))
h = int(os.environ.get('CUBE_H', '4'))
d = 128
torch.manual_seed(240909)
q = (torch.randn(b, t, h, d, device=device) * 0.2).to(torch.bfloat16)
k = (torch.randn_like(q) * 0.2).to(torch.bfloat16)
v = (torch.randn_like(q) * 0.1).to(torch.bfloat16)
g = torch.randn(b, t, h, d, device=device) * 0.1
beta = torch.randn(b, t, h, device=device)
a_log = torch.linspace(-1.0, 0.2, h, device=device)
bias = torch.randn(h, d, device=device) * 0.03
common = dict(A_log=a_log, bias=bias, lower_bound=-1.0,
              initial_state=torch.zeros(b, h, d, d, device=device),
              output_final_state=True)
fns = {
    'persistent': lambda: kda_bt16_fwd_ascendc(q, k, v, g, beta, k2_mode='persistent', **common),
    'triton_aiv': lambda: kda_bt16_fwd_ascendc(q, k, v, g, beta, k2_mode='triton_aiv', **common),
}
for fn in fns.values():
    fn(); torch.npu.synchronize()
rows = {}
for name, fn in fns.items():
    for _ in range(2):
        fn(); torch.npu.synchronize()
    samples = []
    for _ in range(5):
        torch.npu.synchronize(); start = time.perf_counter()
        fn(); torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    rows[name] = {'samples_ms': samples, 'median_ms': samples[len(samples) // 2]}
ref = fns['persistent'](); got = fns['triton_aiv'](); torch.npu.synchronize()
rows['shape'] = [b, t, h, d]
rows['aiv_speedup_vs_persistent'] = rows['persistent']['median_ms'] / rows['triton_aiv']['median_ms']
rows['output_max_abs'] = float((got[0] - ref[0]).abs().max().cpu())
rows['state_max_abs'] = float((got[1] - ref[1]).abs().max().cpu())
rows['finite'] = bool(torch.isfinite(got[0]).all() and torch.isfinite(got[1]).all())
print(json.dumps(rows, indent=2), flush=True)
