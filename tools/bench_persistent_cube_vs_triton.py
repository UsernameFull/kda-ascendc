from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch_npu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'src'))
from kda_ascendc_v1.experimental import get_last_profile, kda_bt16_fwd_ascendc_experimental as kda_bt16_fwd_ascendc
from kda_bt16 import kda_bt16_fwd

DEVICE = torch.device('npu:0')
D = 128


def sync():
    torch.npu.synchronize()


def timed(fn, warmup=2, reps=5):
    for _ in range(warmup):
        fn()
        sync()
    values = []
    for _ in range(reps):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        values.append((time.perf_counter() - start) * 1000)
    values.sort()
    return {'samples_ms': values, 'median_ms': values[len(values) // 2]}


def main():
    torch.npu.set_device(DEVICE)
    torch.manual_seed(240909)
    b = int(os.environ.get('CUBE_B', '2'))
    t = int(os.environ.get('CUBE_T', '1024'))
    h = int(os.environ.get('CUBE_H', '4'))
    q = (torch.randn(b, t, h, D, device=DEVICE) * 0.2).to(torch.bfloat16)
    k = (torch.randn_like(q) * 0.2).to(torch.bfloat16)
    v = (torch.randn_like(q) * 0.1).to(torch.bfloat16)
    g = torch.randn(b, t, h, D, device=DEVICE) * 0.1
    beta = torch.randn(b, t, h, device=DEVICE)
    a_log = torch.linspace(-1.0, 0.2, h, device=DEVICE)
    bias = torch.randn(h, D, device=DEVICE) * 0.03
    initial_state = torch.zeros(b, h, D, D, device=DEVICE)
    common = dict(A_log=a_log, bias=bias, lower_bound=-1.0,
                  initial_state=initial_state, output_final_state=True)
    cube = lambda: kda_bt16_fwd_ascendc(q, k, v, g, beta,
                                         k2_mode='persistent_scan_cube', **common)
    triton = lambda: kda_bt16_fwd(
        q, k, v, g, beta, initial_state=initial_state,
        output_final_state=True, use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
        safe_gate=True, lower_bound=-1.0, A_log=a_log,
        dt_bias=bias.reshape(-1),
    )
    cube()
    sync()
    triton()
    sync()
    cube_t = timed(cube)
    triton_t = timed(triton)
    cube_out, cube_state = cube()
    triton_out, triton_state = triton()
    sync()
    row = {
        'shape': [b, t, h, D],
        'persistent_scan_cube': cube_t,
        'triton': triton_t,
        'triton_speedup_vs_cube': cube_t['median_ms'] / triton_t['median_ms'],
        'cube_speedup_vs_triton': triton_t['median_ms'] / cube_t['median_ms'],
        'output_max_abs': float((cube_out - triton_out).abs().max()),
        'state_max_abs': float((cube_state - triton_state).abs().max()),
        'finite': bool(torch.isfinite(cube_out).all() and torch.isfinite(cube_state).all()),
        'profile': get_last_profile(),
    }
    out = ROOT / 'results/PERSISTENT_SCAN'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'persistent_cube_vs_triton_{b}_{t}_{h}.json').write_text(json.dumps(row, indent=2) + os.linesep)
    print(json.dumps(row, indent=2), flush=True)


if __name__ == '__main__':
    main()
