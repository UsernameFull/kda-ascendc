"""Long-run stability gate: many consecutive calls, bit-identical results.

AscendC cross-pipe / cross-core sync problems do not fail a single call: they
show up as an occasional device fault (507014/507015 aicore errors), as a
warning plus garbage, or as a result that quietly differs between two calls
with the same inputs.  The gate is therefore "N calls in a row, every one
bit-identical to the first, every one finite", on the shapes the pipeline is
tuned for, with the iteration count from ``KDA_STRESS_ITERS`` (default 30).

    python -m pytest tests/test_stability_gate.py -q
    KDA_STRESS_ITERS=100 python -m pytest tests/test_stability_gate.py -q
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
from kda_ascendc_v1.api import (CHUNK, SUPPORTED_CHUNKS, get_last_profile,
                                kda_bt16_fwd_ascendc)

D = 128
ITERS = int(os.environ.get("KDA_STRESS_ITERS", "30"))
# (batch, T, heads); T has to be a multiple of the build's CHUNK.
CORE = (1, 8192, 96)
SIDE = [(1, 2048, 48), (2, 1024, 8)]


def _inputs(b, t, h, device, seed=240921):
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


def _assert_unsupported_build_is_refused():
    """The api has to refuse a known-broken build before compiling anything.

    CPU tensors are enough: the guard sits in front of _check_inputs, so a
    refused build never touches the device or the RTC compile.
    """
    q = torch.zeros(1, 2 * CHUNK, 1, D)
    with pytest.raises(ValueError,
                       match="KDA_CHUNK=%d is an unsupported build" % CHUNK):
        kda_bt16_fwd_ascendc(q, q, q, q.float(), torch.zeros(1, 2 * CHUNK, 1))


def _skip_unless_supported():
    if CHUNK in SUPPORTED_CHUNKS:
        return
    _assert_unsupported_build_is_refused()
    pytest.skip("KDA_CHUNK=%d is an unsupported build (see api.SUPPORTED_CHUNKS)"
                % CHUNK)


def test_unsupported_chunk_build_is_refused():
    if CHUNK in SUPPORTED_CHUNKS:
        pytest.skip("KDA_CHUNK=%d is a supported build" % CHUNK)
    _assert_unsupported_build_is_refused()


def _kw(a_log, bias, initial_state):
    return dict(A_log=a_log, bias=bias, lower_bound=-1.0,
                initial_state=initial_state, output_final_state=True)


@pytest.mark.npu
@pytest.mark.slow
def test_core_shape_survives_repeated_calls():
    _skip_unless_supported()
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    b, t, h = CORE
    q, k, v, g, beta, a_log, bias, initial_state = _inputs(b, t, h, device)
    kw = _kw(a_log, bias, initial_state)
    first_out = first_state = None
    for it in range(ITERS):
        out, state = kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
        torch.npu.synchronize()
        assert bool(torch.isfinite(out).all()), "iteration %d: non-finite output" % it
        assert bool(torch.isfinite(state).all()), "iteration %d: non-finite state" % it
        if it == 0:
            first_out, first_state = out, state
            continue
        assert torch.equal(out, first_out), "iteration %d: output drifted" % it
        assert torch.equal(state, first_state), "iteration %d: state drifted" % it
    counts = get_last_profile()["launch_counts"]
    assert counts.get("kda_k2_persistent_loop") == 1, counts


@pytest.mark.npu
@pytest.mark.slow
@pytest.mark.parametrize("b,t,h", SIDE)
def test_side_shapes_survive_repeated_calls(b, t, h):
    """5 calls each, and the last one still matches the fp32 reference.

    The reference comparison runs after the repeats, so a device that was left
    in a bad state by the loop (rather than by a single call) still fails here.
    """
    from test_torch_reference import torch_reference_kda_bt16

    _skip_unless_supported()
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    q, k, v, g, beta, a_log, bias, initial_state = _inputs(b, t, h, device)
    kw = _kw(a_log, bias, initial_state)
    first_out = None
    for it in range(5):
        out, state = kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
        torch.npu.synchronize()
        assert bool(torch.isfinite(out).all() and torch.isfinite(state).all())
        if it == 0:
            first_out, first_state = out, state
        else:
            assert torch.equal(out, first_out), "iteration %d: output drifted" % it
            assert torch.equal(state, first_state), "iteration %d: state drifted" % it
    qc, kc, vc = (x.float().cpu() for x in (q, k, v))
    ref_out, ref_state = torch_reference_kda_bt16(
        qc, kc, vc, g.cpu(), beta.cpu(), D ** -0.5, lower_bound=-1.0,
        A_log=a_log.cpu(), dt_bias=bias.cpu(),
        initial_state=initial_state.cpu(), state_v_first=True)
    out_rel = float((out.float().cpu() - ref_out.float()).abs().max()) / (
        float(ref_out.abs().max()) + 1e-12)
    state_rel = float((state.cpu() - ref_state.float()).abs().max()) / (
        float(ref_state.abs().max()) + 1e-12)
    assert out_rel < 2e-2, out_rel
    assert state_rel < 2e-2, state_rel
