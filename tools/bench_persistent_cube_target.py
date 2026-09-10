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
from kda_ascendc_v1.api import get_last_profile, kda_bt16_fwd_ascendc

DEVICE = torch.device('npu:0')
D = 128


def sync():
    torch.npu.synchronize()


def timed(fn, warmup=2, reps=5):
    for _ in range(warmup):
        fn()
        sync()
    samples = []
    for _ in range(reps):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    return {'samples_ms': samples, 'median_ms': samples[len(samples) // 2]}


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
    scan = lambda: kda_bt16_fwd_ascendc(q, k, v, g, beta,
                                         k2_mode='persistent_scan', **common)
    cube = lambda: kda_bt16_fwd_ascendc(q, k, v, g, beta,
                                         k2_mode='persistent_scan_cube', **common)
    scan()
    sync()
    cube()
    sync()
    scan_t = timed(scan)
    cube_t = timed(cube)
    scan_out, scan_state = scan()
    cube_out, cube_state = cube()
    sync()
    row = {
        'shape': [b, t, h, D],
        'persistent_scan': scan_t,
        'persistent_scan_cube': cube_t,
        'cube_speedup_vs_scan': scan_t['median_ms'] / cube_t['median_ms'],
        'cube_output_max_abs_vs_scan': float((cube_out - scan_out).abs().max()),
        'cube_state_max_abs_vs_scan': float((cube_state - scan_state).abs().max()),
        'finite': bool(torch.isfinite(cube_out).all() and torch.isfinite(cube_state).all()),
        'profile': get_last_profile(),
    }
    out = ROOT / 'results/PERSISTENT_SCAN'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'persistent_cube_{b}_{t}_{h}.json').write_text(json.dumps(row, indent=2) + os.linesep)
    print(json.dumps(row, indent=2), flush=True)


if __name__ == '__main__':
    main()
