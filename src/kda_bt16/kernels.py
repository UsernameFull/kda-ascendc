# KDA forward Triton kernels (BT=16), Ascend NPU / CUDA portable.
#
# Production path (kda_bt16_kernel_k1/k2, two-stage split):
#   K1: gate/cumsum + intra QK^T + 16x16 forward-substitution solve (via a
#       small fp32 Akk_scratch round-trip, which measured faster than register-only
#       variants on 910B3) + W/U; NT*BH parallel programs.
#   K2: delta-rule state propagation + output, state resident on-chip.
# Experimental (kda_bt16_kernel_fused, use_fused_kernel=True): the same math
#   in a single kernel per (b,hv), with the solve kept register-only and zero
#   intermediate HBM traffic; slower on 910B3 because fusing collapses K1's
#   grid into B*HV serial programs.
#
# Target bucket (matches FlashKDA verifier): bf16, K=V=128, HV==H, safe_gate,
# use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel,
# state_v_first. Fixed-length sequences only.
#
# Reference math (mirrors fla/ops/kda/chunk_fwd.py):
#   g      = cumsum(lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))) * RCP_LN2
#   Aqk    = scale * masked_lower(q*exp2(g-mid) @ (k*exp2(mid-g))^T)
#   A      = (I + Akk_lo)^-1,  Akk_lo = masked_strict_lower(kk^T * beta * exp2(g_s - g_t))
#   w      = A @ (k * beta * exp2(g))
#   u      = A @ (v * beta)
#   v_new  = u - w @ h                      (h: pre-chunk state)
#   o      = scale * (q*exp2(g)) @ h + Aqk @ v_new
#   h      = h * exp2(g_last) + (k*exp2(g_last - g))^T @ v_new
#
# Numerics follow the production path: Aqk/Akk and the 16x16 solve are fp32
# (allow_tf32=False), while the K2 state/output dots are bf16 with fp32
# accumulation, matching chunk_gated_delta_rule_fwd_h / chunk_gla_fwd_o_gk.

import torch
import triton
import triton.language as tl

_WORKSPACE_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}

# RCP_LN2 = 1 / ln(2): gate cumsum is carried in the log2 domain.
_RCP_LN2 = tl.constexpr(1.4426950216)
_EPS = tl.constexpr(1e-6)
_BT = 16


@triton.jit
def _exp(x):
    return tl.exp(x.to(tl.float32))


@triton.jit
def _exp2(x):
    return tl.math.exp2(x.to(tl.float32))


@triton.jit(do_not_specialize=["T", "NT"])
def kda_bt16_kernel_k1(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    Aqk,
    A_inv,
    Akk_scratch,
    w,
    u,
    g_out,
    scale,
    lower_bound,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    NT,
    USE_GATE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
    USE_BETA_SIGMOID: tl.constexpr,
    STORE_A_INV: tl.constexpr,
):
    i_task = tl.program_id(0)
    i_t = i_task % NT
    i_bh = i_task // NT
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    bos = tl.cast(i_b * T, tl.int64)
    i_ti = i_t * BT
    o_c = i_ti + tl.arange(0, BT)
    m_c = o_c < T
    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    m_k = o_k < K

    b_beta = tl.load(beta + (bos * HV + i_hv) + o_c * HV, mask=m_c, other=0.0).to(tl.float32)
    if USE_BETA_SIGMOID:
        b_beta = tl.sigmoid(b_beta)

    p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_ti, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_ti, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * HV + i_hv) * V, (T, V), (HV * V, 1), (i_ti, 0), (BT, BK), (1, 0))
    p_g = tl.make_block_ptr(g + (bos * HV + i_hv) * K, (T, K), (HV * K, 1), (i_ti, 0), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

    if USE_QK_L2NORM:
        b_qf = b_q.to(tl.float32)
        b_q = (b_qf * (1 / tl.sqrt(tl.sum(b_qf * b_qf, 1) + _EPS))[:, None]).to(b_q.dtype)
        b_kf = b_k.to(tl.float32)
        b_k = (b_kf * (1 / tl.sqrt(tl.sum(b_kf * b_kf, 1) + _EPS))[:, None]).to(b_k.dtype)

    if USE_GATE:
        b_A = tl.load(A_log + i_hv).to(tl.float32)
        b_s = b_g
        if HAS_BIAS:
            b_bias = tl.load(dt_bias + i_hv * K + o_k, mask=m_k, other=0.0).to(tl.float32)
            b_s += b_bias[None, :]
        b_gate = lower_bound * tl.sigmoid(_exp(b_A) * b_s)
        b_gate = tl.where(m_c[:, None], b_gate, 0.0)
        b_g = tl.cumsum(b_gate, axis=0) * _RCP_LN2

    p_g_out = tl.make_block_ptr(g_out + (bos * HV + i_hv) * K, (T, K), (HV * K, 1), (i_ti, 0), (BT, BK), (1, 0))
    tl.store(p_g_out, b_g, boundary_check=(0, 1))

    # Gate re-centering: subtract the chunk-local mid-token gate before the
    # exp2, so the QK^T exponents stay near 0 instead of reaching the full
    # chunk cumsum (|BT * lower_bound * RCP_LN2|, which overflows fp32 for
    # |lower_bound| >~ 11 at BT=16). `mid` is a chunk-LOCAL row index, so it
    # must be compared against o_i (local), not o_c (global token index).
    mid = min(BT // 2, T - i_ti - 1)
    b_gn = tl.sum(tl.where(o_i[:, None] == mid, b_g, 0.0), 0)
    b_gm = b_g - b_gn[None, :]

    b_gq = tl.where(m_c[:, None], _exp2(b_gm), 0.0)
    b_gk = tl.where(m_c[:, None], _exp2(-b_gm), 0.0)

    b_kgt = tl.trans(b_k * b_gk)
    b_Aqk = tl.dot(b_q * b_gq, b_kgt, allow_tf32=False) * scale
    b_Akk = tl.dot(b_k * b_gq, b_kgt, allow_tf32=False) * b_beta[:, None]

    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk, 0.0)

    base_bt = (bos * HV + i_hv) * BT
    p_Aqk = tl.make_block_ptr(Aqk + base_bt, (T, BT), (HV * BT, 1), (i_ti, 0), (BT, BT), (1, 0))
    p_Akk_scratch = tl.make_block_ptr(Akk_scratch + base_bt, (T, BT), (HV * BT, 1), (i_ti, 0), (BT, BT), (1, 0))
    tl.store(p_Aqk, b_Aqk.to(p_Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk_scratch, b_Akk, boundary_check=(0, 1))
    tl.debug_barrier()

    # forward substitution solve; BC == BT == 16 -> NC == 1, no inter kernel
    # needed. Measured fastest on 910B3: the 16x16 fp32 Akk_scratch store/load hits
    # L2 and beats register-only alternatives (where+sum row extraction and
    # Neumann-doubling dots were both ~10% slower overall).
    b_Ai = -tl.where(m_Akk, b_Akk, 0.0)
    for i in range(2, min(BT, T - i_ti)):
        b_a = -tl.load(Akk_scratch + ((bos + i_ti + i) * HV + i_hv) * BT + o_i, mask=o_i < BT, other=0.0)
        b_a = tl.where(o_i < i, b_a, 0.0)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where(o_i[:, None] == i, b_a, b_Ai)
    b_Ai += tl.where(o_i[:, None] == o_i[None, :], 1.0, 0.0)

    # w/u (mirror recompute_w_u_fwd_kda_npu)
    b_A = b_Ai.to(b_q.dtype)
    b_kb = (b_k * b_beta[:, None]).to(b_q.dtype)
    b_kb = (b_kb * _exp2(b_g)).to(b_q.dtype)
    b_w = tl.dot(b_A + 0.0, b_kb, allow_tf32=False)
    b_vb = (b_v * b_beta[:, None]).to(b_q.dtype)
    b_u = tl.dot(b_A + 0.0, b_vb, allow_tf32=False)

    p_w = tl.make_block_ptr(w + (bos * HV + i_hv) * K, (T, K), (HV * K, 1), (i_ti, 0), (BT, BK), (1, 0))
    p_u = tl.make_block_ptr(u + (bos * HV + i_hv) * V, (T, V), (HV * V, 1), (i_ti, 0), (BT, BK), (1, 0))
    tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    if STORE_A_INV:
        p_A_inv = tl.make_block_ptr(A_inv + base_bt, (T, BT), (HV * BT, 1), (i_ti, 0), (BT, BT), (1, 0))
        tl.store(p_A_inv, b_Ai.to(p_A_inv.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T", "NT"])
def kda_bt16_kernel_k2(
    q,
    k,
    w,
    u,
    g,
    Aqk,
    o,
    h0,
    ht,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NT,
    NV,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    i_task = tl.program_id(0)
    i_nh = i_task // NV
    i_v = i_task % NV
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.cast(i_n * T, tl.int64)

    o_k = tl.arange(0, K)
    m_k = o_k < K
    v_off = i_v * BV

    if STATE_V_FIRST:
        b_h = tl.zeros([BV, K], dtype=tl.float32)
        if USE_INITIAL_STATE:
            p_h0 = tl.make_block_ptr(h0 + i_nh * K * V, (V, K), (K, 1), (v_off, 0), (BV, K), (1, 0))
            b_h += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_h = tl.zeros([K, BV], dtype=tl.float32)
        if USE_INITIAL_STATE:
            p_h0 = tl.make_block_ptr(h0 + i_nh * K * V, (K, V), (V, 1), (0, v_off), (K, BV), (1, 0))
            b_h += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in tl.range(0, NT, num_stages=NUM_STAGES):
        i_ti = i_t * BT

        p_w = tl.make_block_ptr(w + (bos * HV + i_hv) * K, (T, K), (HV * K, 1), (i_ti, 0), (BT, K), (1, 0))
        p_u = tl.make_block_ptr(u + (bos * HV + i_hv) * V, (T, V), (HV * V, 1), (i_ti, v_off), (BT, BV), (1, 0))
        p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_ti, 0), (BT, K), (1, 0))
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_ti, 0), (BT, K), (1, 0))
        p_g = tl.make_block_ptr(g + (bos * HV + i_hv) * K, (T, K), (HV * K, 1), (i_ti, 0), (BT, K), (1, 0))
        p_A = tl.make_block_ptr(
            Aqk + (bos * HV + i_hv) * BT, (T, BT), (HV * BT, 1), (i_ti, 0), (BT, BT), (1, 0)
        )

        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_u = tl.load(p_u, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_A = tl.load(p_A, boundary_check=(0, 1))

        if USE_QK_L2NORM:
            b_qf = b_q.to(tl.float32)
            b_q = (b_qf * (1 / tl.sqrt(tl.sum(b_qf * b_qf, 1) + _EPS))[:, None]).to(b_q.dtype)
            b_kf = b_k.to(tl.float32)
            b_k = (b_kf * (1 / tl.sqrt(tl.sum(b_kf * b_kf, 1) + _EPS))[:, None]).to(b_k.dtype)

        # v_new = u - w @ h (h: pre-chunk state, bf16 dot, fp32 acc)
        b_hc = b_h.to(b_w.dtype)
        b_hct = tl.trans(b_hc)
        if STATE_V_FIRST:
            b_v = b_u.to(tl.float32) - tl.dot(b_w, b_hct, allow_tf32=False)
        else:
            b_v = b_u.to(tl.float32) - tl.dot(b_w, b_hc, allow_tf32=False)
        b_vb = b_v.to(b_q.dtype)

        # o = scale * (q*exp2(g)) @ h + Aqk @ v_new (bf16 dots)
        b_qg = (b_q.to(tl.float32) * _exp2(b_g)).to(b_q.dtype)
        if STATE_V_FIRST:
            b_o = tl.dot(b_qg, b_hct, allow_tf32=False) * scale
        else:
            b_o = tl.dot(b_qg, b_hc, allow_tf32=False) * scale
        b_o += tl.dot(b_A, b_vb, allow_tf32=False)

        p_o = tl.make_block_ptr(o + (bos * HV + i_hv) * V, (T, V), (HV * V, 1), (i_ti, v_off), (BT, BV), (1, 0))
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        # state update: h = h*exp2(g_last) + (k*exp2(g_last - g))^T @ v_new (bf16 dot)
        last_idx = min(i_ti + BT, T) - 1
        b_g_last = tl.load(
            g + ((bos + last_idx) * HV + i_hv) * K + o_k, mask=m_k, other=0.0
        ).to(tl.float32)
        b_d = _exp2(b_g_last[None, :] - b_g)
        b_kg = (b_k.to(tl.float32) * b_d).to(b_q.dtype)
        b_kgt = tl.trans(b_kg)
        if STATE_V_FIRST:
            b_h = b_h * _exp2(b_g_last)[None, :]
            b_h += tl.trans(tl.dot(b_kgt, b_vb, allow_tf32=False))
        else:
            b_h = b_h * _exp2(b_g_last)[:, None]
            b_h += tl.dot(b_kgt, b_vb, allow_tf32=False)

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            p_ht = tl.make_block_ptr(ht + i_nh * K * V, (V, K), (K, 1), (v_off, 0), (BV, K), (1, 0))
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        else:
            p_ht = tl.make_block_ptr(ht + i_nh * K * V, (K, V), (V, 1), (0, v_off), (K, BV), (1, 0))
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T", "NT"])
def kda_bt16_kernel_fused(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    scale,
    lower_bound,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NT,
    NV,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    USE_GATE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
    USE_BETA_SIGMOID: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # NOTE: per-chunk intermediates (Aqk/A/w/u/g) live only in registers here;
    # kda_bt16_debug() exposes them via the two-stage kernels instead.
    i_task = tl.program_id(0)
    i_nh = i_task // NV
    i_v = i_task % NV
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.cast(i_n * T, tl.int64)

    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, K)
    m_k = o_k < K
    v_off = i_v * BV

    if STATE_V_FIRST:
        b_h = tl.zeros([BV, K], dtype=tl.float32)
        if USE_INITIAL_STATE:
            p_h0 = tl.make_block_ptr(h0 + i_nh * K * V, (V, K), (K, 1), (v_off, 0), (BV, K), (1, 0))
            b_h += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_h = tl.zeros([K, BV], dtype=tl.float32)
        if USE_INITIAL_STATE:
            p_h0 = tl.make_block_ptr(h0 + i_nh * K * V, (K, V), (V, 1), (0, v_off), (K, BV), (1, 0))
            b_h += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in tl.range(0, NT, num_stages=NUM_STAGES):
        i_ti = i_t * BT
        o_c = i_ti + o_i
        m_c = o_c < T

        p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_ti, 0), (BT, K), (1, 0))
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_ti, 0), (BT, K), (1, 0))
        p_v = tl.make_block_ptr(v + (bos * HV + i_hv) * V, (T, V), (HV * V, 1), (i_ti, v_off), (BT, BV), (1, 0))
        p_g = tl.make_block_ptr(g + (bos * HV + i_hv) * K, (T, K), (HV * K, 1), (i_ti, 0), (BT, K), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

        b_beta = tl.load(beta + (bos * HV + i_hv) + o_c * HV, mask=m_c, other=0.0).to(tl.float32)
        if USE_BETA_SIGMOID:
            b_beta = tl.sigmoid(b_beta)

        if USE_QK_L2NORM:
            b_qf = b_q.to(tl.float32)
            b_q = (b_qf * (1 / tl.sqrt(tl.sum(b_qf * b_qf, 1) + _EPS))[:, None]).to(b_q.dtype)
            b_kf = b_k.to(tl.float32)
            b_k = (b_kf * (1 / tl.sqrt(tl.sum(b_kf * b_kf, 1) + _EPS))[:, None]).to(b_k.dtype)

        if USE_GATE:
            b_A = tl.load(A_log + i_hv).to(tl.float32)
            b_s = b_g
            if HAS_BIAS:
                b_bias = tl.load(dt_bias + i_hv * K + o_k, mask=m_k, other=0.0).to(tl.float32)
                b_s += b_bias[None, :]
            b_gate = lower_bound * tl.sigmoid(_exp(b_A) * b_s)
            b_gate = tl.where(m_c[:, None], b_gate, 0.0)
            b_g = tl.cumsum(b_gate, axis=0) * _RCP_LN2

        # 1. intra QK^T (fp32, allow_tf32=False)
        # `mid` is a chunk-LOCAL row index -> compare against o_i, not o_c.
        mid = min(BT // 2, T - i_ti - 1)
        b_gn = tl.sum(tl.where(o_i[:, None] == mid, b_g, 0.0), 0)
        b_gm = b_g - b_gn[None, :]
        b_gq = tl.where(m_c[:, None], _exp2(b_gm), 0.0)
        b_gk = tl.where(m_c[:, None], _exp2(-b_gm), 0.0)
        b_kgt = tl.trans(b_k * b_gk)
        b_Aqk = tl.dot(b_q * b_gq, b_kgt, allow_tf32=False) * scale
        b_Akk = tl.dot(b_k * b_gq, b_kgt, allow_tf32=False) * b_beta[:, None]

        m_Aqk = o_i[:, None] >= o_i[None, :]
        m_Akk = o_i[:, None] > o_i[None, :]
        b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
        b_Akk = tl.where(m_Akk, b_Akk, 0.0)

        # 2. register-only forward-substitution solve (no Akk_scratch global round-trip)
        b_Ai = -b_Akk
        for i in range(2, min(BT, T - i_ti)):
            b_a = tl.sum(tl.where(o_i[:, None] == i, b_Akk, 0.0), 0)
            b_a = -tl.where(o_i < i, b_a, 0.0)
            b_a += tl.sum(b_a[:, None] * b_Ai, 0)
            b_Ai = tl.where(o_i[:, None] == i, b_a, b_Ai)
        b_Ai += tl.where(o_i[:, None] == o_i[None, :], 1.0, 0.0)

        # 3. w/u (bf16 quantized; mirrors recompute_w_u_fwd_kda_npu)
        b_Abf = b_Ai.to(b_q.dtype)
        b_kb = (b_k * b_beta[:, None]).to(b_q.dtype)
        b_kb = (b_kb * _exp2(b_g)).to(b_q.dtype)
        b_w = tl.dot(b_Abf + 0.0, b_kb, allow_tf32=False)
        b_vb = (b_v * b_beta[:, None]).to(b_q.dtype)
        b_u = tl.dot(b_Abf + 0.0, b_vb, allow_tf32=False)

        # 4. K2 recurrence: v_new / output / state update (bf16 dots)
        b_wq = b_w.to(b_q.dtype)
        b_uq = b_u.to(b_q.dtype)
        b_hc = b_h.to(b_wq.dtype)
        b_hct = tl.trans(b_hc)
        if STATE_V_FIRST:
            b_v = b_uq.to(tl.float32) - tl.dot(b_wq, b_hct, allow_tf32=False)
        else:
            b_v = b_uq.to(tl.float32) - tl.dot(b_wq, b_hc, allow_tf32=False)
        b_vb = b_v.to(b_q.dtype)

        b_Aqk_q = b_Aqk.to(b_q.dtype)
        b_qg = (b_q.to(tl.float32) * _exp2(b_g)).to(b_q.dtype)
        if STATE_V_FIRST:
            b_o = tl.dot(b_qg, b_hct, allow_tf32=False) * scale
        else:
            b_o = tl.dot(b_qg, b_hc, allow_tf32=False) * scale
        b_o += tl.dot(b_Aqk_q, b_vb, allow_tf32=False)

        p_o = tl.make_block_ptr(o + (bos * HV + i_hv) * V, (T, V), (HV * V, 1), (i_ti, v_off), (BT, BV), (1, 0))
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        # state update: h = h*exp2(g_last) + (k*exp2(g_last - g))^T @ v_new
        # g_last comes from the in-register cumsum (the `g` pointer holds the
        # raw gate input here, unlike the two-stage K2 which reads g_cumsum).
        last_idx = min(i_ti + BT, T) - 1
        b_g_last = tl.sum(tl.where(o_c[:, None] == last_idx, b_g, 0.0), 0)
        b_d = _exp2(b_g_last[None, :] - b_g)
        b_kg = (b_k.to(tl.float32) * b_d).to(b_q.dtype)
        b_kgt = tl.trans(b_kg)
        if STATE_V_FIRST:
            b_h = b_h * _exp2(b_g_last)[None, :]
            b_h += tl.trans(tl.dot(b_kgt, b_vb, allow_tf32=False))
        else:
            b_h = b_h * _exp2(b_g_last)[:, None]
            b_h += tl.dot(b_kgt, b_vb, allow_tf32=False)

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            p_ht = tl.make_block_ptr(ht + i_nh * K * V, (V, K), (K, 1), (v_off, 0), (BV, K), (1, 0))
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        else:
            p_ht = tl.make_block_ptr(ht + i_nh * K * V, (K, V), (V, 1), (0, v_off), (K, BV), (1, 0))
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def _run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    lower_bound: float,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    state_v_first: bool,
    return_intermediates: bool,
    use_fused: bool = False,
    reuse_workspace: bool = False,
):
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    kernel_state_v_first = state_v_first
    kernel_initial_state = initial_state
    if not state_v_first:
        # Execute the validated V-first layout and transpose only at the API
        # boundary. This avoids the NPU-only mismatch in the separate K,V
        # branch when an initial state is supplied.
        kernel_state_v_first = True
        if initial_state is not None:
            kernel_initial_state = initial_state.transpose(-1, -2).contiguous()

    def restore_state_layout(state: torch.Tensor | None) -> torch.Tensor | None:
        if state is not None and not state_v_first:
            return state.transpose(-1, -2).contiguous()
        return state

    BT = _BT
    # Keep the full state resident, but use a smaller V tile on Ascend.  The
    # 910B/A3 UB is 192 KiB and the K2 path materializes several fp32
    # intermediates around the resident state.  With BV=V=128 the compiler
    # needs about 256 KiB and rejects the kernel with UB overflow.  Splitting
    # V into two 64-wide programs keeps the same math and output layout while
    # bringing each program below the UB limit.  CUDA keeps the original
    # single-program fast path.
    BV = 64 if q.device.type == "npu" and V == 128 else V
    NV = triton.cdiv(V, BV)

    NT = triton.cdiv(T, BT)

    cache_key = (q.device, B, T, H, HV, K, V, k.dtype, return_intermediates)
    workspace = _WORKSPACE_CACHE.get(cache_key) if reuse_workspace else None

    def alloc(name, shape, dtype):
        if workspace is not None:
            tensor = workspace.get(name)
            if tensor is not None and tensor.shape == shape and tensor.dtype == dtype:
                return tensor
        tensor = torch.empty(shape, dtype=dtype, device=q.device)
        if reuse_workspace:
            _WORKSPACE_CACHE.setdefault(cache_key, {})[name] = tensor
        return tensor

    o = alloc("o", (B, T, HV, V), k.dtype)
    if output_final_state:
        if kernel_state_v_first:
            ht = alloc("ht", (B, HV, V, K), torch.float32)
        else:
            ht = alloc("ht", (B, HV, K, V), torch.float32)
    else:
        ht = None

    if use_fused and not return_intermediates:
        # experimental single-kernel path; K1 math (gate/Aqk/solve/w/u) runs
        # per chunk inside the state loop and intermediates never touch HBM.
        # Measured slower than the two-stage split on 910B3 (~1.5x at large T):
        # fusing collapses K1's NT*BH-program grid into B*HV serial programs,
        # which loses far more from lost parallelism than it gains from the
        # removed barrier + intermediate round-trips. Kept as reference.
        kda_bt16_kernel_fused[(B * HV * NV,)](
            q, k, v, g, beta, A_log, dt_bias,
            o, kernel_initial_state, ht, scale, lower_bound, T,
            H=H, HV=HV, K=K, V=V, BT=BT, BV=BV,
            NT=NT, NV=NV,
            USE_INITIAL_STATE=kernel_initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
            STATE_V_FIRST=kernel_state_v_first,
            USE_GATE=use_gate_in_kernel,
            HAS_BIAS=dt_bias is not None,
            USE_QK_L2NORM=use_qk_l2norm_in_kernel,
            USE_BETA_SIGMOID=use_beta_sigmoid_in_kernel,
            NUM_STAGES=1,
        )
        return o, restore_state_layout(ht)

    # production path: two-stage kernels. K1 runs with NT*BH programs (high
    # grid parallelism), K2 keeps the state resident per (b, hv).
    BH = B * HV
    Aqk = alloc("Aqk", (B, T, HV, BT), k.dtype)
    # P2: A_inv only needed for debug/intermediates, not read by K2
    A_inv = alloc("A_inv", (B, T, HV, BT), k.dtype) if return_intermediates else None
    Akk_scratch = alloc("Akk_scratch", (B, T, HV, BT), torch.float32)
    w = alloc("w", (B, T, HV, K), k.dtype)
    u = alloc("u", (B, T, HV, V), k.dtype)
    g_cumsum = alloc("g_cumsum", (B, T, HV, K), torch.float32)

    kda_bt16_kernel_k1[(NT * BH,)](
        q, k, v, g, beta, A_log, dt_bias,
        Aqk, A_inv, Akk_scratch, w, u, g_cumsum,
        scale, lower_bound, T,
        H=H, HV=HV, K=K, V=V, BT=BT, BK=K,
        NT=NT,
        USE_GATE=use_gate_in_kernel,
        HAS_BIAS=dt_bias is not None,
        USE_QK_L2NORM=use_qk_l2norm_in_kernel,
        USE_BETA_SIGMOID=use_beta_sigmoid_in_kernel,
        STORE_A_INV=return_intermediates,
    )
    kda_bt16_kernel_k2[(B * HV * NV,)](
        q, k, w, u, g_cumsum, Aqk, o, kernel_initial_state, ht, scale, T,
        H=H, HV=HV, K=K, V=V, BT=BT, BV=BV,
        NT=NT, NV=NV,
        USE_INITIAL_STATE=kernel_initial_state is not None,
        STORE_FINAL_STATE=output_final_state,
        STATE_V_FIRST=kernel_state_v_first,
        USE_QK_L2NORM=use_qk_l2norm_in_kernel,
        NUM_STAGES=1,
    )

    if return_intermediates:
        return o, restore_state_layout(ht), {"Aqk": Aqk, "A_inv": A_inv, "w": w, "u": u, "g": g_cumsum}
    return o, restore_state_layout(ht)


def _validate_kda_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    use_gate_in_kernel: bool,
    safe_gate: bool,
    lower_bound: float | None,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    state_v_first: bool,
) -> tuple[int, int, int, int, int, int]:
    """
    Shared input validation for kda_bt16_fwd and kda_bt16_debug.
    
    Returns:
        (B, T, H, K, HV, V) shape tuple
    
    Raises:
        ValueError/TypeError for invalid inputs
    """
    # P1-9: ndim validation (before unpacking to avoid confusing Python errors)
    if q.ndim != 4:
        raise ValueError(f"q must be 4D [B, T, H, D], got {q.ndim}D with shape {q.shape}")
    if k.ndim != 4:
        raise ValueError(f"k must be 4D [B, T, H, D], got {k.ndim}D with shape {k.shape}")
    if v.ndim != 4:
        raise ValueError(f"v must be 4D [B, T, HV, V], got {v.ndim}D with shape {v.shape}")
    if g.ndim != 4:
        raise ValueError(f"g must be 4D [B, T, HV, D], got {g.ndim}D with shape {g.shape}")
    if beta.ndim != 3:
        raise ValueError(f"beta must be 3D [B, T, HV], got {beta.ndim}D with shape {beta.shape}")
    
    # Shape extraction
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    
    # P1-9: Empty shape validation
    if B <= 0 or T <= 0 or H <= 0 or K <= 0:
        raise ValueError(f"q dimensions must be positive, got B={B}, T={T}, H={H}, K={K}")
    if HV <= 0 or V <= 0:
        raise ValueError(f"v dimensions must be positive, got HV={HV}, V={V}")
    
    # P0: Contiguous check - kernels hardcode strides
    if not q.is_contiguous():
        raise ValueError("q must be contiguous (kernel assumes stride [H*K, K, 1])")
    if not k.is_contiguous():
        raise ValueError("k must be contiguous")
    if not v.is_contiguous():
        raise ValueError("v must be contiguous")
    if not g.is_contiguous():
        raise ValueError("g must be contiguous")
    if not beta.is_contiguous():
        raise ValueError("beta must be contiguous")
    
    # P1: Dtype validation
    if q.dtype != torch.bfloat16:
        raise TypeError(f"q must be bfloat16, got {q.dtype}")
    if k.dtype != torch.bfloat16:
        raise TypeError(f"k must be bfloat16, got {k.dtype}")
    if v.dtype != torch.bfloat16:
        raise TypeError(f"v must be bfloat16, got {v.dtype}")
    if use_gate_in_kernel and g.dtype != torch.float32:
        raise TypeError(f"g must be float32 when use_gate_in_kernel=True, got {g.dtype}")
    if beta.dtype != torch.float32:
        raise TypeError(f"beta must be float32, got {beta.dtype}")
    
    # P1: Shape validation
    if k.shape != q.shape:
        raise ValueError(f"k.shape {k.shape} must match q.shape {q.shape}")
    if v.shape != (B, T, HV, V):
        raise ValueError(f"v.shape {v.shape} must be [B={B}, T={T}, HV={HV}, V={V}]")
    if g.shape != (B, T, HV, K):
        raise ValueError(f"g.shape {g.shape} must be [B={B}, T={T}, HV={HV}, K={K}]")
    if beta.shape != (B, T, HV):
        raise ValueError(f"beta.shape {beta.shape} must be [B={B}, T={T}, HV={HV}]")
    
    # P1: Device validation
    if not all(x.device == q.device for x in [k, v, g, beta]):
        raise ValueError(f"All tensors must be on same device as q ({q.device})")
    
    # Initial state validation
    if initial_state is not None:
        if not initial_state.is_contiguous():
            raise ValueError("initial_state must be contiguous")
        if initial_state.dtype != torch.float32:
            raise TypeError(f"initial_state must be float32, got {initial_state.dtype}")
        if initial_state.device != q.device:
            raise ValueError(f"initial_state device {initial_state.device} must match q ({q.device})")
        expected_shape = (B, HV, V, K) if state_v_first else (B, HV, K, V)
        if initial_state.shape != expected_shape:
            raise ValueError(f"initial_state.shape {initial_state.shape} must be {expected_shape} (state_v_first={state_v_first})")
    
    # A_log validation
    if A_log is not None:
        if not A_log.is_contiguous():
            raise ValueError("A_log must be contiguous")
        if A_log.dtype != torch.float32:
            raise TypeError(f"A_log must be float32, got {A_log.dtype}")
        if A_log.device != q.device:
            raise ValueError(f"A_log device must match q")
        if A_log.shape != (HV,):
            raise ValueError(f"A_log.shape {A_log.shape} must be (HV={HV},)")
    
    # dt_bias validation
    if dt_bias is not None:
        if not dt_bias.is_contiguous():
            raise ValueError("dt_bias must be contiguous")
        if dt_bias.dtype != torch.float32:
            raise TypeError(f"dt_bias must be float32, got {dt_bias.dtype}")
        if dt_bias.device != q.device:
            raise ValueError(f"dt_bias device must match q")
        if dt_bias.shape != (HV * K,):
            raise ValueError(f"dt_bias.shape {dt_bias.shape} must be (HV*K={HV*K},)")
    
    # Prototype constraints
    if K != 128 or V != 128:
        raise ValueError(f"prototype targets K=V=128, got K={K}, V={V}")
    if HV != H:
        raise ValueError(f"prototype targets MHA (HV == H), got HV={HV}, H={H}")
    
    # P2: safe_gate parameter validation
    if use_gate_in_kernel:
        if not safe_gate:
            raise ValueError("use_gate_in_kernel requires safe_gate=True")
        if lower_bound is None:
            raise ValueError("use_gate_in_kernel requires lower_bound")
        # Check range to prevent fp32 exp2 overflow
        # At BT=16, gate recentering can amplify exponent by ~8x
        # Safe range for fp32 exp2: |lower_bound| < 11 to avoid overflow
        if not (-11.0 < lower_bound < 0.0):
            raise ValueError(
                f"lower_bound must be in range (-11, 0) to prevent fp32 exp2 overflow "
                f"(BT=16 can amplify exponent by ~8x), got {lower_bound}"
            )
        if A_log is None:
            raise ValueError("use_gate_in_kernel requires A_log")
    
    return B, T, H, K, HV, V


def kda_bt16_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    safe_gate: bool = True,
    lower_bound: float = -5.0,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    state_v_first: bool = True,
    use_fused_kernel: bool = False,
    reuse_workspace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """KDA forward.

    Default path is the two-stage kernel split (K1 intra + K2 state), which
    maximizes grid parallelism on Ascend 910B. Set ``use_fused_kernel=True``
    for the experimental single-kernel variant (zero intermediate HBM
    traffic; measured ~1.5x slower at large T on 910B3 due to lost K1 grid
    parallelism).

    Args:
        q/k: ``[B, T, H, K]``, bf16, ``K == 128``.
        v: ``[B, T, HV, V]``, bf16, ``HV == H``, ``V == 128``.
        g: raw gate ``[B, T, HV, K]`` fp32 if ``use_gate_in_kernel``, else the
            pre-computed chunk-local cumsum in the log2 domain.
        beta: ``[B, T, HV]``; raw logits if ``use_beta_sigmoid_in_kernel``.
        initial_state: fp32 ``[B, HV, V, K]`` (``state_v_first=True``) or
            ``[B, HV, K, V]``.

    Returns:
        ``o`` of shape ``[B, T, HV, V]`` and, if ``output_final_state``,
        the fp32 final state.
    """
    # Use shared validation
    B, T, H, K, HV, V = _validate_kda_inputs(
        q, k, v, g, beta, initial_state,
        use_gate_in_kernel, safe_gate, lower_bound,
        A_log, dt_bias, state_v_first
    )
    
    if scale is None:
        scale = K ** -0.5

    return _run(
        q, k, v, g, beta, scale, initial_state, output_final_state,
        use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel,
        lower_bound, A_log, dt_bias, state_v_first,
        return_intermediates=False,
        use_fused=use_fused_kernel,
        reuse_workspace=reuse_workspace,
    )


def kda_bt16_debug(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    safe_gate: bool = True,
    lower_bound: float = -5.0,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    state_v_first: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
    """Like :func:`kda_bt16_fwd` but runs the two-stage kernels and also
    returns per-chunk intermediates ``{Aqk, A_inv, w, u, g}`` for stage-wise
    numerical validation."""
    # Use shared validation
    B, T, H, K, HV, V = _validate_kda_inputs(
        q, k, v, g, beta, initial_state,
        use_gate_in_kernel, safe_gate, lower_bound,
        A_log, dt_bias, state_v_first
    )
    if scale is None:
        scale = K ** -0.5

    return _run(
        q, k, v, g, beta, scale, initial_state, output_final_state,
        use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel,
        lower_bound, A_log, dt_bias, state_v_first,
        return_intermediates=True,
        use_fused=False,
        reuse_workspace=False,
    )
