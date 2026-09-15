"""CHUNK x shape correctness matrix for the public KDA entry point.

Every build of the pipeline (C=16/32/64) has to reproduce the same fp32 KDA:
the chunked recurrence is chunk-size invariant in exact arithmetic, so a
disagreement means the kernels are solving a different problem than the one the
host thinks they are (the C=64 build once walked 16-row pieces of its 64-row
chunks - fast and wrong).

The chunk size is a compile-time constant of the process, so the matrix is run
once per build:

    python -m pytest tests/test_chunk_shape_matrix.py -q                 # C=16
    KDA_CHUNK=32 python -m pytest tests/test_chunk_shape_matrix.py -q    # C=32
    KDA_CHUNK=64 python -m pytest tests/test_chunk_shape_matrix.py -q    # C=64

Each case runs the public entry point (default ``k2_mode``) twice for
determinism and once against the host fp64/fp32 reference.
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
# (batch, heads, chunks, with_initial_state).  The chunk counts are short
# (2 chunks) for the full B x H cross and long (8 chunks) for the heads on
# either end of the range, which is where a head-map or block-count bug shows.
MATRIX = [
    (1, 2, 2, True),
    (1, 32, 2, False),
    (1, 48, 2, True),
    (1, 96, 2, False),
    (2, 2, 2, True),
    (2, 32, 2, True),
    (2, 48, 2, False),
    (2, 96, 2, False),
    (1, 2, 8, False),
    (1, 32, 8, True),
]


def _inputs(b, t, h, device, seed):
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
                       match="KDA_CHUNK=%d is a known-broken" % CHUNK):
        kda_bt16_fwd_ascendc(q, q, q, q.float(), torch.zeros(1, 2 * CHUNK, 1))


def _skip_unless_supported():
    if CHUNK in SUPPORTED_CHUNKS:
        return
    _assert_unsupported_build_is_refused()
    pytest.skip("KDA_CHUNK=%d is a known-broken build (see api.SUPPORTED_CHUNKS)"
                % CHUNK)


def test_unsupported_chunk_build_is_refused():
    if CHUNK in SUPPORTED_CHUNKS:
        pytest.skip("KDA_CHUNK=%d is a supported build" % CHUNK)
    _assert_unsupported_build_is_refused()


def _relative(err, ref):
    return err / (float(ref.abs().max()) + 1e-12)


@pytest.mark.npu
@pytest.mark.parametrize("b,h,chunks,with_state", MATRIX)
def test_public_path_matches_fp32_reference(b, h, chunks, with_state):
    from test_torch_reference import torch_reference_kda_bt16

    _skip_unless_supported()
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    t = chunks * CHUNK
    q, k, v, g, beta, a_log, bias, initial_state = _inputs(
        b, t, h, device, seed=240920 + 1000 * b + 10 * h + chunks)
    kw = dict(A_log=a_log, bias=bias, lower_bound=-1.0,
              output_final_state=True,
              initial_state=initial_state if with_state else None)
    out, state = kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    torch.npu.synchronize()
    counts = get_last_profile()["launch_counts"]
    assert counts.get("kda_k2_persistent_loop") == 1, counts
    assert bool(torch.isfinite(out).all() and torch.isfinite(state).all())
    assert out.shape == (b, t, h, D) and out.dtype == torch.bfloat16
    assert state.shape == (b, h, D, D) and state.dtype == torch.float32

    again_out, again_state = kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    torch.npu.synchronize()
    assert torch.equal(out, again_out), "output is not deterministic across runs"
    assert torch.equal(state, again_state), "state is not deterministic across runs"

    qc, kc, vc = (x.float().cpu() for x in (q, k, v))
    ref_out, ref_state = torch_reference_kda_bt16(
        qc, kc, vc, g.cpu(), beta.cpu(), D ** -0.5, lower_bound=-1.0,
        A_log=a_log.cpu(), dt_bias=bias.cpu(),
        initial_state=initial_state.cpu() if with_state else None,
        state_v_first=True)
    out_err = float((out.float().cpu() - ref_out.float()).abs().max())
    state_err = float((state.cpu() - ref_state.float()).abs().max())
    out_rel = _relative(out_err, ref_out)
    state_rel = _relative(state_err, ref_state)
    assert out_rel < 2e-2, (out_rel, out_err)
    assert state_rel < 2e-2, (state_rel, state_err)
