"""The public ``[B, T, H, D]`` output must not be axis-swapped.

The modes that returned through ``out_task`` used to hand back a tensor whose
memory was laid out ``[b, h, t, d]`` while it was shaped ``[b, t, h, d]``.  The
mode-vs-mode comparisons could not see that (every mode was swapped the same
way) and the reference comparison accepted it because the swap error is the
same size as the signal - 4.7e-3 against a 4.6e-3 reference absmax.  Pin the
two properties that actually distinguish the layouts: the error against the
Triton reference has to be a small fraction of the signal, and it has to be
smaller than the error of the ``(t, h)`` swapped candidate.
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "src"))

from kda_bt16 import kda_bt16_fwd  # noqa: E402
from kda_ascendc_v1.api import kda_bt16_fwd_ascendc  # noqa: E402


@pytest.mark.npu
def test_output_axes_are_not_swapped():
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    b, t, h, d = 1, 64, 4, 128
    torch.manual_seed(1212)
    q = (torch.randn(b, t, h, d, device=device) * 0.2).to(torch.bfloat16)
    k = (torch.randn_like(q) * 0.2).to(torch.bfloat16)
    v = (torch.randn_like(q) * 0.1).to(torch.bfloat16)
    g = torch.randn(b, t, h, d, device=device) * 0.1
    beta = torch.randn(b, t, h, device=device)
    a_log = torch.linspace(-1.0, 0.2, h, device=device)
    bias = torch.randn(h, d, device=device) * 0.03
    initial_state = torch.randn(b, h, d, d, device=device) * 0.01
    common = dict(A_log=a_log, bias=bias, lower_bound=-1.0,
                  initial_state=initial_state, output_final_state=True)

    # Ground truth: the Triton reference in src/kda_bt16, so the check does not
    # compare two of our own modes against each other.
    ref, _ = kda_bt16_fwd(
        q, k, v, g, beta, initial_state=initial_state, output_final_state=True,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True, safe_gate=True, lower_bound=-1.0,
        A_log=a_log, dt_bias=bias.reshape(-1))
    torch.npu.synchronize()
    ref_f = ref.float()
    signal = float(ref_f.abs().max())
    ref_swapped = ref_f.transpose(1, 2)  # [b, h, t, d] view of the same values
    assert signal > 0.0

    for mode in ("separated", "cube_separated", "cube_d3_separated", "cube_full_d4",
                 "mix_aic_1_2", "mix_d12_vnew", "persistent_scan_cube", "triton_aiv"):
        out, _ = kda_bt16_fwd_ascendc(q, k, v, g, beta, k2_mode=mode, **common)
        torch.npu.synchronize()
        out_f = out.float()
        err = float((out_f - ref_f).abs().max()) / signal
        swapped = float((out_f.view(b, h, t, d) - ref_swapped).abs().max()) / signal
        assert err < 0.05, f"{mode}: relative output error {err:.4e}"
        assert err < 0.5 * swapped, (
            f"{mode}: the [t, h] swapped candidate matches better ({swapped:.4e}) "
            f"than the direct comparison ({err:.4e}); the output axes are swapped")
