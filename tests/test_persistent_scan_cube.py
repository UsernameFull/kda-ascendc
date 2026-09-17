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
from kda_ascendc_v1.experimental import get_last_profile, kda_bt16_fwd_ascendc_experimental as kda_bt16_fwd_ascendc

from kda_ascendc_v1.api import CHUNK
if CHUNK != 16:
    pytest.skip("C=16-only experiment mode: the persistent_scan kernels carry "
                "M = 16 as a literal (api.C16_ONLY_K2_MODES), so this build's "
                "KDA_CHUNK=%d cannot run them" % CHUNK, allow_module_level=True)


@pytest.mark.npu
def test_persistent_scan_cube_uses_cube_stages():
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    b, t, h, d = 1, 16, 2, 128
    torch.manual_seed(240909)
    q = (torch.randn(b, t, h, d, device=device) * 0.2).to(torch.bfloat16)
    k = (torch.randn(b, t, h, d, device=device) * 0.2).to(torch.bfloat16)
    v = (torch.randn(b, t, h, d, device=device) * 0.1).to(torch.bfloat16)
    g = torch.randn(b, t, h, d, device=device) * 0.1
    beta = torch.randn(b, t, h, device=device)
    a_log = torch.linspace(-1.0, 0.2, h, device=device)
    bias = torch.randn(h, d, device=device) * 0.03
    initial_state = torch.zeros(b, h, d, d, device=device)
    out, state = kda_bt16_fwd_ascendc(
        q, k, v, g, beta, A_log=a_log, bias=bias,
        lower_bound=-1.0, initial_state=initial_state,
        output_final_state=True, k2_mode="persistent_scan_cube",
    )
    torch.npu.synchronize()
    profile = get_last_profile()
    assert bool(torch.isfinite(out).all() and torch.isfinite(state).all())
    assert profile["launch_counts"].get("kda_k2_mix_all_cube") == 1

