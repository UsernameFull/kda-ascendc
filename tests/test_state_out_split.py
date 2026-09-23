"""The route-1 split (``k2_mode="split_state_out"``) against the fused loop.

``kernels/v1/k2_state_loop.cpp`` + ``kernels/v1/k2_out_parallel.cpp`` are the
2026-09-22 redesign's first judgment candidate: the serial chain keeps
Z = U - W S and S = diag(d) S + Kg^T Z and publishes the chunk-entry state
H[c], and a second, fully parallel kernel computes O[c] = Q[c] H[c] + A[c] Z[c].

The measurement that decided against it lives in the plan (section 11.30: the
pair is +1.70 ms, +16.1%, at [1,8192,96,128]/C=64).  What this file pins is the
part that must not rot if the candidate is ever revisited: the split reproduces
the fused loop *bit for bit* (every rounding position is the same - the Cube
still rounds d2/d3 to bf16 and the vector side still accumulates
out = d2*scale + d3 in fp32), at every chunk size the build supports.

The mode is reachable only through ``kda_ascendc_v1.experimental``: it is not
the shipped path, so the public entry point has to keep refusing it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "kernels" / "v1" / "k2_state_loop.cpp").exists():
    pytest.skip("AscendC v1 sources are not present", allow_module_level=True)
sys.path.insert(0, str(ROOT / "python"))

import kda_ascendc_v1.api as api  # noqa: E402
from kda_ascendc_v1.api import (CHUNK, SPLIT_STATE_OUT,  # noqa: E402
                                get_last_profile, kda_bt16_fwd_ascendc)
from kda_ascendc_v1.experimental import (  # noqa: E402
    kda_bt16_fwd_ascendc_experimental)

pytestmark = pytest.mark.npu
D = 128


def _skip_without_npu():
    if not torch.npu.is_available() or torch.npu.device_count() == 0:
        pytest.skip("no NPU device")


def _inputs(b, t, h, dev, seed=20260922):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=gen).to(device=dev, dtype=dtype)

    q = randn(b, t, h, D, dtype=torch.bfloat16)
    k = randn(b, t, h, D, dtype=torch.bfloat16)
    v = randn(b, t, h, D, dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(randn(b, t, h, D)).clamp_min(-5.0).contiguous()
    beta = randn(b, t, h).sigmoid()
    return q, k, v, g, beta, randn(h), randn(h, D) * 0.1


@pytest.mark.parametrize("b,h,chunks", [(1, 2, 8), (2, 3, 4)])
def test_split_is_bit_identical_to_the_fused_loop(b, h, chunks):
    """Same K1, same inputs, two K2s: the outputs and the final state must match
    to the bit, at the chunk size this process was built with."""
    _skip_without_npu()
    dev = torch.device("npu", 0)
    t = chunks * CHUNK
    q, k, v, g, beta, a_log, bias = _inputs(b, t, h, dev)
    kw = dict(A_log=a_log, bias=bias, lower_bound=-5.0, output_final_state=True)
    o_f, s_f = kda_bt16_fwd_ascendc(q, k, v, g, beta, **kw)
    o_s, s_s = kda_bt16_fwd_ascendc_experimental(
        q, k, v, g, beta, k2_mode=SPLIT_STATE_OUT, **kw)
    torch.npu.synchronize()
    assert torch.equal(o_f, o_s), (
        "out differs: max %.3e" % (o_f.float() - o_s.float()).abs().max().item())
    assert torch.equal(s_f, s_s), (
        "final_state differs: max %.3e"
        % (s_f.float() - s_s.float()).abs().max().item())


def test_the_split_is_two_launches_and_the_public_api_refuses_it():
    """One state launch, one output launch, and no way in through the public
    entry point: the candidate must stay experimental while it loses."""
    _skip_without_npu()
    dev = torch.device("npu", 0)
    t = 4 * CHUNK
    q, k, v, g, beta, a_log, bias = _inputs(1, t, 2, dev)
    kw = dict(A_log=a_log, bias=bias, lower_bound=-5.0, output_final_state=True)
    kda_bt16_fwd_ascendc_experimental(q, k, v, g, beta,
                                      k2_mode=SPLIT_STATE_OUT, **kw)
    torch.npu.synchronize()
    counts = dict(get_last_profile()["launch_counts"])
    assert counts.get("kda_k2_state_loop") == 1
    assert counts.get("kda_k2_out_parallel") == 1
    assert counts.get("kda_k2_persistent_loop") is None
    with pytest.raises(ValueError, match="unsupported k2_mode"):
        kda_bt16_fwd_ascendc(q, k, v, g, beta, k2_mode=SPLIT_STATE_OUT, **kw)
    assert SPLIT_STATE_OUT not in api.C16_ONLY_K2_MODES
    assert SPLIT_STATE_OUT in api.K2_MODES
