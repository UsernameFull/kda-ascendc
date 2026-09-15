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


def sync() -> None:
    torch.npu.synchronize()


def timed(fn, warmup: int = 2, reps: int = 5) -> dict[str, object]:
    for _ in range(warmup):
        fn()
        sync()
    samples = []
    for _ in range(reps):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    return {"samples_ms": samples, "median_ms": samples[len(samples) // 2]}


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
    common = dict(A_log=a_log, bias=bias, lower_bound=-1.0,
                  initial_state=initial_state, output_final_state=True,
                  k2_mode="mix_d12_vnew")
    regular = lambda: kda_bt16_fwd_ascendc(q, k, v, g, beta, **common)

    regular()
    sync()
    regular_timing = timed(regular)
    regular_out, regular_state = regular()
    sync()
    regular_profile = get_last_profile()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_result = regular()
    sync()
    graph_timing = timed(graph.replay)
    graph_out, graph_state = graph_result
    sync()

    row = {
        "shape": [b, t, h, D],
        "regular": regular_timing,
        "graph_replay": graph_timing,
        "graph_speedup": regular_timing["median_ms"] / graph_timing["median_ms"],
        "output_max_abs": float((graph_out - regular_out).abs().cpu().max()),
        "state_max_abs": float((graph_state - regular_state).abs().cpu().max()),
        "finite": bool(torch.isfinite(graph_out).all() and torch.isfinite(graph_state).all()),
        "capture_profile": get_last_profile(),
        "regular_profile": regular_profile,
    }
    output_dir = ROOT / "results" / "GRAPH"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"bench_graph_{b}_{t}_{h}.json"
    output_file.write_text(json.dumps(row, indent=2) + os.linesep)
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
