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
from kda_ascendc_v1.api import CHUNK, get_last_profile, kda_bt16_fwd_ascendc
from kda_ascendc_v1.experimental import kda_bt16_fwd_ascendc_experimental


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
@pytest.mark.parametrize("chunks,h", [(1, 2), (4, 2), (2, 2)])
def test_default_mode_matches_separated_oracle(chunks, h):
    """The public entry point defaults to the device-side chunk loop.

    Called without ``k2_mode``, it must agree with the per-chunk path.

    ``k2_mode="separated"`` is a C=16 implementation (its kernels carry the
    16-row tile as a literal) and lives in ``kda_ascendc_v1.experimental``, so
    this oracle only exists for a C=16 build.
    C=32/64 builds are covered by
    ``test_persistent_loop_matches_fp32_reference``, which is chunk-generic.
    """
    if CHUNK != 16:
        pytest.skip("the per-chunk path is a C=16-only oracle")
    b, t = 1, chunks * CHUNK
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    q, k, v, g, beta, a_log, bias, initial_state = _inputs(b, t, h, 128, device)
    kw = dict(A_log=a_log, bias=bias, lower_bound=-1.0,
              initial_state=initial_state, output_final_state=True)
    ref_out, ref_state = kda_bt16_fwd_ascendc_experimental(
        q, k, v, g, beta, k2_mode="separated", **kw)
    torch.npu.synchronize()
    # No k2_mode: this is what a caller of the public API gets.
    out, state = kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
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
    # 48 heads is what makes every configuration put more than one head in a
    # block (the api caps MAXH at ceil(bh / 24) = 2 at 48 heads, at any chunk
    # size), which is the configuration the depth-one flag protocol is built
    # for: a missed drain has to survive the head interleave to show up.
    q, k, v, g, beta, a_log, bias, initial_state = _inputs(1, 512, 48, 128, device, seed=240912)
    kw = dict(A_log=a_log, bias=bias, lower_bound=-1.0,
              initial_state=initial_state, output_final_state=True)
    first_out, first_state = kda_bt16_fwd_ascendc(q, k, v, g, beta, k2_mode="persistent_loop", **kw)
    torch.npu.synchronize()
    for _ in range(3):
        out, state = kda_bt16_fwd_ascendc(q, k, v, g, beta, k2_mode="persistent_loop", **kw)
        torch.npu.synchronize()
        assert torch.equal(out, first_out)
        assert torch.equal(state, first_state)


@pytest.mark.npu
@pytest.mark.parametrize("chunks,h", [(2, 2), (4, 48)])
def test_persistent_loop_matches_fp32_reference(chunks, h):
    """Chunk-generic gate: the device recurrence must land on the fp32 KDA.

    The chunked recurrence is chunk-size invariant in exact arithmetic, so a
    build at any KDA_CHUNK (16/32/64) has to reproduce the same fp32 reference
    within the bf16 rounding envelope.  This is the test that covers a C=64
    build - the kernel used to walk 16-row pieces of 64-row chunks there, which
    is fast and wrong, and no other case in this file would have caught it.
    """
    from test_torch_reference import torch_reference_kda_bt16

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    b, t = 1, chunks * CHUNK
    q, k, v, g, beta, a_log, bias, initial_state = _inputs(b, t, h, 128, device, seed=240913)
    out, state = kda_bt16_fwd_ascendc(
        q, k, v, g, beta, A_log=a_log, bias=bias, lower_bound=-1.0,
        initial_state=initial_state, output_final_state=True,
        k2_mode="persistent_loop")
    torch.npu.synchronize()
    profile = get_last_profile()
    assert profile["launch_counts"].get("kda_k2_persistent_loop") == 1
    assert profile["launch_counts"].get("kda_k2_d12_kernel", 0) == 0

    # The reference always chunks at BT=16 and solves in fp64 on the host.
    qc, kc, vc = (x.float().cpu() for x in (q, k, v))
    ref_out, ref_state = torch_reference_kda_bt16(
        qc, kc, vc, g.cpu(), beta.cpu(), 128 ** -0.5, lower_bound=-1.0,
        A_log=a_log.cpu(), dt_bias=bias.cpu(),
        initial_state=initial_state.cpu(), state_v_first=True)
    out_err = float((out.float().cpu() - ref_out.float()).abs().max())
    state_err = float((state.float().cpu() - ref_state.float()).abs().max())
    out_rel = out_err / (float(ref_out.abs().max()) + 1e-12)
    state_rel = state_err / (float(ref_state.abs().max()) + 1e-12)
    assert out_rel < 2e-2, (out_rel, out_err)
    assert state_rel < 2e-2, (state_rel, state_err)
