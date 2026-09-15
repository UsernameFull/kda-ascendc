from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch_npu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from kda_ascendc_v1.experimental import get_last_profile, kda_bt16_fwd_ascendc_experimental as kda_bt16_fwd_ascendc

D = 128
DEVICE = torch.device("npu:0")


def sync():
    torch.npu.synchronize()


def timed(fn, warmup=2, reps=3):
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
    return {"samples_ms": samples, "median_ms": samples[len(samples) // 2]}


def main():
    torch.npu.set_device(DEVICE)
    torch.manual_seed(1313)
    b = int(os.environ.get("CUBE_B", "1"))
    t = int(os.environ.get("CUBE_T", "32"))
    h = int(os.environ.get("CUBE_H", "2"))
    q = (torch.randn(b, t, h, D, device=DEVICE) * 0.2).to(torch.bfloat16)
    k = (torch.randn(b, t, h, D, device=DEVICE) * 0.2).to(torch.bfloat16)
    v = (torch.randn(b, t, h, D, device=DEVICE) * 0.1).to(torch.bfloat16)
    g = torch.randn(b, t, h, D, device=DEVICE) * 0.1
    beta = torch.randn(b, t, h, device=DEVICE)
    a_log = torch.linspace(-1.0, 0.2, h, device=DEVICE)
    bias = torch.zeros(h, D, device=DEVICE)
    state = torch.zeros(b, h, D, D, device=DEVICE)
    common = dict(A_log=a_log, bias=bias, lower_bound=-1.0,
                  initial_state=state, output_final_state=True)
    funcs = {
        "persistent": lambda: kda_bt16_fwd_ascendc(q, k, v, g, beta,
                                                     k2_mode="persistent", **common),
        "persistent_scan": lambda: kda_bt16_fwd_ascendc(q, k, v, g, beta,
                                                          k2_mode="persistent_scan", **common),
    }
    for fn in funcs.values():
        fn()
        sync()
    timings = {name: timed(fn) for name, fn in funcs.items()}
    values = {name: fn() for name, fn in funcs.items()}
    sync()
    baseline_out, baseline_state = values["persistent"]
    scan_out, scan_state = values["persistent_scan"]
    scan_profile = get_last_profile()
    row = {
        "shape": [b, t, h, D],
        **timings,
        "persistent_scan_speedup": timings["persistent"]["median_ms"] / timings["persistent_scan"]["median_ms"],
        "output_max_abs": float((scan_out - baseline_out).abs().max().cpu()),
        "state_max_abs": float((scan_state - baseline_state).abs().max().cpu()),
        "scan_profile": scan_profile,
        "finite": bool(torch.isfinite(scan_out).all() and torch.isfinite(scan_state).all()),
    }
    output_dir = ROOT / "results" / "PERSISTENT_SCAN"
    output_dir.mkdir(exist_ok=True)
    (output_dir / f"bench_{b}_{t}_{h}.json").write_text(json.dumps(row, indent=2) + os.linesep)
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
