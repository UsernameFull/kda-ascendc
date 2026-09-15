"""The public API serves one K2 implementation; the rest is experimental.

``k2_mode`` used to default to ``"separated"``, a C=16 kernel: a caller of the
public entry point at any other chunk size got a 16-row answer that looks fast
and is wrong.  These tests pin the contract: the default is the chunk-generic
device-side loop, the historical modes are only reachable through
``kda_ascendc_v1.experimental``, and that entry refuses them unless the build
really is C=16.
"""

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
from kda_ascendc_v1.api import (CHUNK, C16_ONLY_K2_MODES, PERSISTENT_LOOP,
                                get_last_profile, kda_bt16_fwd_ascendc)
from kda_ascendc_v1.experimental import kda_bt16_fwd_ascendc_experimental

D = 128


def _inputs(b, t, h, device, seed=240915):
    torch.manual_seed(seed)
    q = (torch.randn(b, t, h, D, device=device) * 0.2).to(torch.bfloat16)
    k = (torch.randn(b, t, h, D, device=device) * 0.2).to(torch.bfloat16)
    v = (torch.randn(b, t, h, D, device=device) * 0.1).to(torch.bfloat16)
    g = torch.randn(b, t, h, D, device=device) * 0.1
    beta = torch.randn(b, t, h, device=device)
    a_log = torch.linspace(-1.0, 0.2, h, device=device)
    bias = torch.randn(h, D, device=device) * 0.03
    initial_state = torch.randn(b, h, D, D, device=device) * 0.01
    return q, k, v, g, beta, a_log, bias, initial_state


def test_public_default_is_the_recommended_mode():
    """Pin the default: ``None`` means "the recommended implementation"."""
    import inspect

    public = inspect.signature(kda_bt16_fwd_ascendc).parameters["k2_mode"].default
    assert public is None, public
    assert PERSISTENT_LOOP not in C16_ONLY_K2_MODES


@pytest.mark.parametrize("mode", sorted(C16_ONLY_K2_MODES))
def test_public_api_rejects_experimental_modes(mode):
    """The mode check runs before any device work, so CPU tensors are enough."""
    q = torch.zeros(1, CHUNK, 1, D)
    with pytest.raises(ValueError, match="experimental"):
        kda_bt16_fwd_ascendc(q, q, q, q.float(), torch.zeros(1, CHUNK, 1), k2_mode=mode)


def test_public_api_rejects_unknown_mode():
    q = torch.zeros(1, CHUNK, 1, D)
    with pytest.raises(ValueError, match="unsupported k2_mode"):
        kda_bt16_fwd_ascendc(q, q, q, q.float(), torch.zeros(1, CHUNK, 1), k2_mode="nope")


def test_experimental_api_rejects_unknown_mode():
    q = torch.zeros(1, CHUNK, 1, D)
    with pytest.raises(ValueError, match="unsupported k2_mode"):
        kda_bt16_fwd_ascendc_experimental(
            q, q, q, q.float(), torch.zeros(1, CHUNK, 1), k2_mode="nope")


@pytest.mark.npu
def test_default_is_the_device_side_chunk_loop():
    """No ``k2_mode`` means one persistent_loop launch and no per-chunk launch."""
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    q, k, v, g, beta, a_log, bias, initial_state = _inputs(1, 2 * CHUNK, 2, device)
    kw = dict(A_log=a_log, bias=bias, lower_bound=-1.0,
              initial_state=initial_state, output_final_state=True)
    out, state = kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    torch.npu.synchronize()
    counts = get_last_profile()["launch_counts"]
    assert counts.get("kda_k2_persistent_loop") == 1, counts
    assert counts.get("kda_k2_d12_kernel", 0) == 0, counts
    assert counts.get("kda_k2_init_kernel", 0) == 0, counts
    assert counts.get("kda_kg_transpose", 0) == 0, counts
    assert bool(torch.isfinite(out).all() and torch.isfinite(state).all())

    explicit_out, explicit_state = kda_bt16_fwd_ascendc(
        q, k, v, g, beta, k2_mode=PERSISTENT_LOOP, **kw)
    torch.npu.synchronize()
    assert torch.equal(out, explicit_out)
    assert torch.equal(state, explicit_state)


@pytest.mark.npu
def test_experimental_entry_serves_the_chunk_generic_loop_too():
    """The C=16-free modes stay bit-identical across the two entry points."""
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    q, k, v, g, beta, a_log, bias, initial_state = _inputs(1, 2 * CHUNK, 2, device)
    kw = dict(A_log=a_log, bias=bias, lower_bound=-1.0,
              initial_state=initial_state, output_final_state=True)
    out, state = kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    torch.npu.synchronize()
    exp_out, exp_state = kda_bt16_fwd_ascendc_experimental(q, k, v, g, beta, **kw)
    torch.npu.synchronize()
    assert torch.equal(out, exp_out)
    assert torch.equal(state, exp_state)


@pytest.mark.npu
def test_experimental_c16_modes_are_gated_on_the_build():
    """A C=16-only kernel is legal at KDA_CHUNK=16 and refused everywhere else."""
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    q, k, v, g, beta, a_log, bias, initial_state = _inputs(1, 2 * CHUNK, 2, device)
    kw = dict(A_log=a_log, bias=bias, lower_bound=-1.0,
              initial_state=initial_state, output_final_state=True)
    if CHUNK != 16:
        with pytest.raises(ValueError, match="KDA_CHUNK=%d" % CHUNK):
            kda_bt16_fwd_ascendc_experimental(
                q, k, v, g, beta, k2_mode="persistent", **kw)
        return
    # Numerics of these modes are covered by their own tests; here only the
    # gate matters.
    out, state = kda_bt16_fwd_ascendc_experimental(
        q, k, v, g, beta, k2_mode="persistent", **kw)
    torch.npu.synchronize()
    counts = get_last_profile()["launch_counts"]
    assert counts.get("kda_k2_persistent_kernel") == 1, counts
    assert bool(torch.isfinite(out).all() and torch.isfinite(state).all())
