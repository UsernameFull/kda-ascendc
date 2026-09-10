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

torch.npu.set_device(0)
DEVICE = torch.device("npu:0")
D = 128


def check(b, t, h):
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
    baseline = kda_bt16_fwd_ascendc(q, k, v, g, beta,
                                    k2_mode="persistent", **common)
    torch.npu.synchronize()
    scanned = kda_bt16_fwd_ascendc(q, k, v, g, beta,
                                   k2_mode="persistent_scan", **common)
    torch.npu.synchronize()
    out_error = float((scanned[0] - baseline[0]).abs().max().cpu())
    state_error = float((scanned[1] - baseline[1]).abs().max().cpu())
    profile = get_last_profile()
    assert out_error == 0.0 and state_error == 0.0
    assert profile["launch_counts"].get("kda_k2_persistent_scan_kernel") == 1
    assert bool(torch.isfinite(scanned[0]).all() and torch.isfinite(scanned[1]).all())


@pytest.mark.npu
def test_persistent_scan_small():
    check(1, 16, 2)


@pytest.mark.npu
def test_persistent_scan_cross_chunk():
    check(1, 64, 2)


@pytest.mark.npu
def test_persistent_scan_many_heads():
    check(1, 16, 32)
