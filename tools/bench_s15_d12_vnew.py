from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch_npu

ROOT = Path("/workspace/kda_ascendc_luna_20260905")
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "baseline/triton_bv64/src"))

from kda_ascendc_v1.api import get_last_profile, kda_bt16_fwd_ascendc
from kernels import kda_bt16_fwd

D = 128
DEVICE = torch.device("npu:0")


def sync() -> None:
    torch.npu.synchronize()


def timed(fn, warmup: int = 2, reps: int = 3) -> dict[str, object]:
    for _ in range(warmup):
        fn()
        sync()
    values = []
    for _ in range(reps):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        values.append((time.perf_counter() - start) * 1e3)
    values.sort()
    return {"samples_ms": values, "median_ms": values[len(values) // 2]}


def main() -> None:
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
    initial_state = torch.zeros(b, h, D, D, device=DEVICE)
    common = dict(
        A_log=a_log,
        bias=bias,
        lower_bound=-1.0,
        initial_state=initial_state,
        output_final_state=True,
    )
    funcs = {
        "cube_full_d4": lambda: kda_bt16_fwd_ascendc(
            q, k, v, g, beta, k2_mode="cube_full_d4", **common
        ),
        "mix_aic_1_2": lambda: kda_bt16_fwd_ascendc(
            q, k, v, g, beta, k2_mode="mix_aic_1_2", **common
        ),
        "mix_d12_vnew": lambda: kda_bt16_fwd_ascendc(
            q, k, v, g, beta, k2_mode="mix_d12_vnew", **common
        ),
        "triton": lambda: kda_bt16_fwd(
            q, k, v, g, beta, initial_state=initial_state,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
            safe_gate=True, lower_bound=-1.0, A_log=a_log,
            dt_bias=bias.reshape(-1),
        ),
    }
    for fn in funcs.values():
        fn()
        sync()
    timings = {}
    profiles = {}
    for name, fn in funcs.items():
        timings[name] = timed(fn)
        if name != "triton" and os.environ.get("KDA_PROFILE", "0") == "1":
            fn()
            sync()
            profiles[name] = get_last_profile()
    values = {name: fn() for name, fn in funcs.items()}
    sync()
    mix_out, mix_state = values["mix_d12_vnew"]
    base_out, base_state = values["cube_full_d4"]
    row = {
        "shape": [b, t, h, D],
        **timings,
        "profiles": profiles,
        "s15_speedup_vs_cube_full": timings["cube_full_d4"]["median_ms"] / timings["mix_d12_vnew"]["median_ms"],
        "s15_speedup_vs_s14_mix": timings["mix_aic_1_2"]["median_ms"] / timings["mix_d12_vnew"]["median_ms"],
        "s15_speedup_vs_triton": timings["triton"]["median_ms"] / timings["mix_d12_vnew"]["median_ms"],
        "s15_output_max_abs_vs_cube_full": float((mix_out - base_out).abs().max().cpu()),
        "s15_state_max_abs_vs_cube_full": float((mix_state - base_state).abs().max().cpu()),
        "finite": bool(torch.isfinite(mix_out).all() and torch.isfinite(mix_state).all()),
    }
    out_dir = ROOT / "results/S15"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"bench_d12_vnew_{b}_{t}_{h}.json").write_text(json.dumps(row, indent=2) + os.linesep)
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
