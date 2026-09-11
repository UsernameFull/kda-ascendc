import os
import sys
from pathlib import Path

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu", reason="Ascend NPU runtime is required")
ROOT = Path(os.environ.get("KDA_ASCENDC_ROOT", Path(__file__).resolve().parents[1]))
if not (ROOT / "python" / "kda_ascendc_v1").exists():
    pytest.skip("AscendC v1 sources are not present", allow_module_level=True)
sys.path.insert(0, str(ROOT / "python"))
from kda_ascendc_v1.api import get_last_profile, kda_bt16_fwd_ascendc


def _inputs(b, t, h, d, device, seed=240911):
    torch.manual_seed(seed)
    q = (torch.randn(b, t, h, d, device=device) * 0.2).to(torch.bfloat16)
    k = (torch.randn(b, t, h, d, device=device) * 0.2).to(torch.bfloat16)
    v = (torch.randn(b, t, h, d, device=device) * 0.1).to(torch.bfloat16)
    g = torch.randn(b, t, h, d, device=device) * 0.1
    beta = torch.randn(b, t, h, device=device)
    a_log = torch.linspace(-1.0, 0.2, h, device=device)
    bias = torch.randn(h, d, device=device) * 0.03
    initial_state = (torch.randn(b, h, d, d, device=device) * 0.01).to(torch.float32)
    return (q, k, v, g, beta, a_log, bias, initial_state)


@pytest.mark.npu
@pytest.mark.parametrize("b,t,h", [(1, 16, 2), (1, 64, 2), (1, 32, 2)])
def test_persistent_loop_matches_separated(b, t, h):
    """The single-launch device-side chunk loop must agree with the per-chunk path."""
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    q, k, v, g, beta, a_log, bias, initial_state = _inputs(b, t, h, 128, device)
    kw = dict(A_log=a_log, bias=bias, lower_bound=-1.0,
              initial_state=initial_state, output_final_state=True)
    ref_out, ref_state = kda_bt16_fwd_ascendc(q, k, v, g, beta, k2_mode="separated", **kw)
    torch.npu.synchronize()
    out, state = kda_bt16_fwd_ascendc(q, k, v, g, beta, k2_mode="persistent_loop", **kw)
    torch.npu.synchronize()
    profile = get_last_profile()
    assert profile["launch_counts"].get("kda_k2_persistent_loop") == 1
    assert profile["launch_counts"].get("kda_k2_d12_kernel", 0) == 0
    assert bool(torch.isfinite(out).all() and torch.isfinite(state).all())
    out_err = float((out.float() - ref_out.float()).abs().max().cpu())
    state_err = float((state.float() - ref_state.float()).abs().max().cpu())
    assert out_err < 1e-3, out_err
    assert state_err < 1e-4, state_err


@pytest.mark.npu
def test_persistent_loop_is_deterministic_across_launches():
    """Every iteration reuses the same UB staging buffers, so a missed
    write-after-read drain shows up as a launch-order-dependent output."""
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    q, k, v, g, beta, a_log, bias, initial_state = _inputs(1, 512, 8, 128, device, seed=240912)
    kw = dict(A_log=a_log, bias=bias, lower_bound=-1.0,
              initial_state=initial_state, output_final_state=True)
    first_out, first_state = kda_bt16_fwd_ascendc(q, k, v, g, beta, k2_mode="persistent_loop", **kw)
    torch.npu.synchronize()
    for _ in range(3):
        out, state = kda_bt16_fwd_ascendc(q, k, v, g, beta, k2_mode="persistent_loop", **kw)
        torch.npu.synchronize()
        assert torch.equal(out, first_out)
        assert torch.equal(state, first_state)
