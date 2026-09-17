"""Correctness checks for the S15 d12/vnew MIX kernel."""

import os
import sys
from pathlib import Path

import pytest
import torch
torch_npu = pytest.importorskip("torch_npu", reason="Ascend NPU runtime is required")

ROOT = Path(os.environ.get("KDA_ASCENDC_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "python"))
if not (ROOT / "python" / "kda_ascendc_v1").exists():
    pytest.skip("AscendC v1 sources are not present", allow_module_level=True)
from kda_ascendc_v1.api import CHUNK
if CHUNK != 16:
    pytest.skip("C=16-only experiment mode: these kernels hard-code a 16-row tile "
                "and the api refuses them at KDA_CHUNK=%d" % CHUNK,
                allow_module_level=True)


torch.npu.set_device(0)
DEVICE = torch.device("npu:0")


def check(b: int, t: int, h: int) -> tuple[float, float]:
    torch.manual_seed(9100 + b + t + h)
    q = (torch.randn(b, t, h, 128, device=DEVICE) * 0.2).to(torch.bfloat16)
    k = (torch.randn_like(q) * 0.2).to(torch.bfloat16)
    v = (torch.randn_like(q) * 0.1).to(torch.bfloat16)
    g = torch.randn(b, t, h, 128, device=DEVICE) * 0.1
    beta = torch.randn(b, t, h, device=DEVICE)
    a_log = torch.linspace(-1.0, 0.2, h, device=DEVICE)
    bias = torch.randn(h, 128, device=DEVICE) * 0.03
    initial_state = torch.randn(b, h, 128, 128, device=DEVICE) * 0.01
    common = dict(A_log=a_log, bias=bias, lower_bound=-1.0,
                  initial_state=initial_state, output_final_state=True)
    from kda_ascendc_v1.experimental import kda_bt16_fwd_ascendc_experimental as kda_bt16_fwd_ascendc

    out, state, dbg = kda_bt16_fwd_ascendc(
        q, k, v, g, beta, k2_mode="mix_d12_vnew",
        return_intermediates=True, **common)
    torch.npu.synchronize()
    ref_out, ref_state, ref_dbg = kda_bt16_fwd_ascendc(
        q, k, v, g, beta, k2_mode="cube_full_d4",
        return_intermediates=True, **common)
    torch.npu.synchronize()
    assert bool(torch.isfinite(out).all() and torch.isfinite(state).all())
    for name in ("d1", "d2", "Vnew", "VnewT", "d3"):
        assert bool(torch.isfinite(dbg[name]).all())
        assert float((dbg[name] - ref_dbg[name]).abs().max().cpu()) == 0.0
    e_out = float((out - ref_out).abs().max().cpu())
    e_state = float((state - ref_state).abs().max().cpu())
    assert e_out == 0.0 and e_state == 0.0, (e_out, e_state)
    return e_out, e_state


print({"t16_h2": check(1, 16, 2),
       "t64_h2": check(1, 64, 2),
       "t16_h32": check(1, 16, 32),
       "status": "passed"})
