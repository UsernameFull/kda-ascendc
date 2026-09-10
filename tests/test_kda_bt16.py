# Correctness test for the two-stage BT=16 KDA kernels vs:
#   1) fused_recurrent_kda (mathematical gold)
#   2) FLA chunk_kda (production Triton path, chunk_size=64)
#
# Requires a checkout of flash-linear-attention (main) for the reference
# implementations; tests are skipped when it is unavailable.
# Run on NPU:  ASCEND_RT_VISIBLE_DEVICES=0 python -m pytest tests/ -x -q

import pytest

fla = pytest.importorskip("fla", reason="flash-linear-attention (main) required for gold references")

import torch
import torch.nn.functional as F  # noqa: F401

from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.utils import assert_close

from kda_bt16 import kda_bt16_fwd

LOWER_BOUND = -5.0


def _gold(q, k, v, g, beta, scale, A_log, dt_bias, h0, state_v_first, lower_bound=LOWER_BOUND):
    return fused_recurrent_kda(
        q, k, v, g, beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=lower_bound,
        state_v_first=state_v_first,
    )


def _make_inputs(B, T, H, HV, D, with_initial_state, dev="npu", seed=42):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, D, dtype=torch.float32, device=dev)
    k = torch.randn(B, T, H, D, dtype=torch.float32, device=dev)
    v = torch.randn(B, T, HV, D, dtype=torch.float32, device=dev)
    g = torch.randn(B, T, HV, D, dtype=torch.float32, device=dev)
    beta = torch.randn(B, T, HV, dtype=torch.float32, device=dev)
    A_log = torch.randn(HV, dtype=torch.float32, device=dev)
    dt_bias = torch.randn(HV * D, dtype=torch.float32, device=dev) * 0.1
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32, device=dev) if with_initial_state else None
    return q, k, v, g, beta, A_log, dt_bias, h0


def _run_case(B, T, H, HV, D, scale, state_v_first, with_initial_state, lower_bound=LOWER_BOUND):
    dev = "npu"
    q, k, v, g, beta, A_log, dt_bias, h0 = _make_inputs(B, T, H, HV, D, with_initial_state, dev)

    # gold: fp32 recurrent
    o_gold, ht_gold = _gold(q, k, v, g, beta, scale, A_log, dt_bias, h0, state_v_first, lower_bound)

    # prototype: bf16, fused gate in K1
    o_p, ht_p = kda_bt16_fwd(
        q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
        g, beta,  # g raw + beta logits stay fp32 for gate/sigmoid path
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=lower_bound,
        A_log=A_log,
        dt_bias=dt_bias,
        state_v_first=state_v_first,
    )

    assert_close("o", o_gold, o_p.float(), 0.006)
    assert_close("ht", ht_gold, ht_p, 0.006)

    # cross-check vs FLA chunk_kda (production path, chunk_size=64).
    # chunk_kda rejects lower_bound outside its safe range [-5, 0).
    if T >= 64 and -5 <= lower_bound < 0:
        o_c, ht_c = chunk_kda(
            q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
            g, beta,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=h0,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=lower_bound,
            state_v_first=state_v_first,
            chunk_size=64,
        )
        assert_close("o_c", o_c.float(), o_p.float(), 0.01)
        assert_close("ht_c", ht_c, ht_p, 0.01)


def test_bt16_basic():
    _run_case(B=2, T=64, H=4, HV=4, D=128, scale=0.1, state_v_first=True, with_initial_state=False)


def test_bt16_with_state():
    _run_case(B=2, T=64, H=4, HV=4, D=128, scale=0.1, state_v_first=True, with_initial_state=True)


def test_bt16_partial_last_chunk():
    _run_case(B=2, T=1037, H=4, HV=4, D=128, scale=0.1, state_v_first=True, with_initial_state=True)


def test_bt16_more_heads():
    _run_case(B=1, T=512, H=8, HV=8, D=128, scale=0.05, state_v_first=True, with_initial_state=True)


def test_bt16_default_scale():
    _run_case(B=2, T=128, H=4, HV=4, D=128, scale=None, state_v_first=True, with_initial_state=False)


def test_bt16_state_kv_layout():
    _run_case(B=2, T=256, H=4, HV=4, D=128, scale=0.1, state_v_first=False, with_initial_state=True)


@pytest.mark.parametrize("lower_bound", [-8.0, -10.0])
def test_bt16_saturated_gate_no_overflow(lower_bound):
    """K1 re-centers the gate cumsum on the chunk-local mid token, which halves
    the worst-case QK^T exponent from |BT * lb * RCP_LN2| to |BT/2 * lb *
    RCP_LN2| (fp32 exp2 overflows past 128). If the re-centering silently
    no-ops -- e.g. by matching `mid` against a global instead of a chunk-local
    row index -- every chunk past the first carries the full cumsum and
    saturating gates blow up to inf/NaN.

    Setup: A_log=0, dt_bias=0, g large => sigmoid(...) ~ 1 => the per-token
    gate pins to `lower_bound`, maximizing the cumsum. T > BT is required
    because only chunks with i_t >= 1 are affected.
    """
    dev = "npu"
    B, T, H, HV, D, scale = 1, 128, 4, 4, 128, 0.1
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=torch.float32, device=dev)
    k = torch.randn(B, T, H, D, dtype=torch.float32, device=dev)
    v = torch.randn(B, T, HV, D, dtype=torch.float32, device=dev)
    beta = torch.randn(B, T, HV, dtype=torch.float32, device=dev)
    A_log = torch.zeros(HV, dtype=torch.float32, device=dev)
    dt_bias = torch.zeros(HV * D, dtype=torch.float32, device=dev)
    g = torch.full((B, T, HV, D), 20.0, dtype=torch.float32, device=dev)

    o_gold, ht_gold = _gold(q, k, v, g, beta, scale, A_log, dt_bias, None, True, lower_bound)

    o_p, ht_p = kda_bt16_fwd(
        q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
        g, beta,
        scale=scale,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=lower_bound,
        A_log=A_log,
        dt_bias=dt_bias,
        state_v_first=True,
    )

    assert torch.isfinite(o_p.float()).all(), "o has inf/NaN: gate re-centering is not working"
    assert torch.isfinite(ht_p).all(), "ht has inf/NaN: gate re-centering is not working"
    assert_close("o", o_gold, o_p.float(), 0.02)
    assert_close("ht", ht_gold, ht_p, 0.02)


@pytest.mark.parametrize("state_v_first", [True, False])
def test_bt16_fused_kernel_matches_two_stage(state_v_first):
    """The experimental single-kernel path must produce the same o AND ht as
    the two-stage split for both state layouts (ht is torch.empty, so a
    missing store surfaces as garbage rather than zeros)."""
    dev = "npu"
    q, k, v, g, beta, A_log, dt_bias, h0 = _make_inputs(
        B=2, T=64, H=4, HV=4, D=128, with_initial_state=True, dev=dev
    )
    kwargs = dict(
        scale=0.1,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=LOWER_BOUND,
        A_log=A_log,
        dt_bias=dt_bias,
        state_v_first=state_v_first,
    )
    qb, kb, vb = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
    o_ref, ht_ref = kda_bt16_fwd(qb, kb, vb, g, beta, **kwargs)
    o_f, ht_f = kda_bt16_fwd(qb, kb, vb, g, beta, use_fused_kernel=True, **kwargs)

    assert_close("o_fused", o_ref.float(), o_f.float(), 0.002)
    assert_close("ht_fused", ht_ref, ht_f, 0.002)


def test_bt16_precomputed_g():
    torch.manual_seed(42)
    dev = "npu"
    B, T, H, HV, D = 2, 128, 4, 4, 128
    q = torch.randn(B, T, H, D, dtype=torch.float32, device=dev)
    k = torch.randn(B, T, H, D, dtype=torch.float32, device=dev)
    v = torch.randn(B, T, HV, D, dtype=torch.float32, device=dev)
    g = torch.randn(B, T, HV, D, dtype=torch.float32, device=dev)
    beta = torch.randn(B, T, HV, dtype=torch.float32, device=dev)
    A_log = torch.randn(HV, dtype=torch.float32, device=dev)
    dt_bias = torch.randn(HV * D, dtype=torch.float32, device=dev) * 0.1
    scale = 0.1

    g_cum = kda_gate_chunk_cumsum(
        g, A_log, chunk_size=16, scale=RCP_LN2, dt_bias=dt_bias, lower_bound=LOWER_BOUND
    )

    o_gold, ht_gold = _gold(q, k, v, g, beta, scale, A_log, dt_bias, None, True)

    o_p, ht_p = kda_bt16_fwd(
        q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
        g_cum, beta,
        scale=scale,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=LOWER_BOUND,
        A_log=A_log,
        dt_bias=dt_bias,
        state_v_first=True,
    )

    assert_close("o", o_gold, o_p.float(), 0.006)
    assert_close("ht", ht_gold, ht_p, 0.006)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-x", "-q"]))