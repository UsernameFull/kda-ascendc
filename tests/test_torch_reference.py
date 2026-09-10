"""
Independent PyTorch reference test for KDA BT=16 kernels.

This test does NOT depend on FLA, ensuring that correctness verification
can run even if FLA is not installed (avoiding silent test skips in CI).

Uses a pure PyTorch reference implementation matching the exact KDA BT=16
mathematical formulation with chunk-wise processing.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kda_bt16 import kda_bt16_fwd


def torch_reference_kda_bt16(
    q, k, v, g, beta, scale, 
    lower_bound=-5.0, 
    A_log=None, 
    dt_bias=None,
    initial_state=None,
    state_v_first=True
):
    """
    Pure PyTorch reference implementation of KDA BT=16.
    
    FP32 mathematical reference matching kernel equations (not bitwise-equivalent).
    
    Equations:
    1. L2 normalization: q_norm = q / sqrt(sum(q²) + eps), ||q_norm|| = 1
    2. Gate processing: cumsum with A_log/dt_bias transformation
    3. Chunk-wise (BT=16) processing:
       - Gate recentering around mid token
       - Aqk, Akk with exp2(g_recentered) scaling
       - A = (I + Akk)^{-1} via forward substitution
       - w = A @ (k * beta * exp2(g))
       - u = A @ (v * beta)
       
       When state_v_first=True (h is [V, K]):
         - v_new = u - w @ h.T
         - o_chunk = (q * exp2(g)) @ h.T * scale + Aqk @ v_new
         - h_new = h * exp2(g_last)[None, :] + v_new.T @ (k * exp2(g_last - g))
       
       When state_v_first=False (h is [K, V]):
         - v_new = u - w @ h
         - o_chunk = (q * exp2(g)) @ h * scale + Aqk @ v_new
         - h_new = h * exp2(g_last)[:, None] + (k * exp2(g_last - g)).T @ v_new
    
    Args:
        q, k, v: [B, T, H, D] tensors
        g: [B, T, H, D] gate (fp32)
        beta: [B, T, H] (fp32)
        scale: float
        lower_bound: float (for gate transformation)
        A_log: [H] optional (exp scaling for gate)
        dt_bias: [H*D] optional (additive bias for gate input)
        initial_state: [B, H, V, K] or [B, H, K, V] optional
        state_v_first: bool (state layout: True=[V,K], False=[K,V])
        
    Returns:
        o: [B, T, H, D] output
        final_state: [B, H, V, K] or [B, H, K, V]
    """
    B, T, H, D = q.shape
    BT = 16
    device = q.device
    dtype = q.dtype
    eps = 1e-6
    
    # 1. L2 normalization (kernel match: ||q|| = 1, no sqrt(D) scaling)
    q_norm = q.float() / torch.sqrt(torch.sum(q.float() ** 2, dim=-1, keepdim=True) + eps)
    k_norm = k.float() / torch.sqrt(torch.sum(k.float() ** 2, dim=-1, keepdim=True) + eps)
    q_norm = q_norm.to(dtype)
    k_norm = k_norm.to(dtype)
    
    # 2. Beta sigmoid
    beta_s = torch.sigmoid(beta)
    
    # 3. Gate transformation
    RCP_LN2 = 1.44269504089
    if A_log is not None:
        A_exp = torch.exp(A_log)  # [H]
        gate_input = g  # [B, T, H, D]
        
        if dt_bias is not None:
            dt_bias_reshaped = dt_bias.view(H, D)  # [H, D]
            gate_input = gate_input + dt_bias_reshaped[None, None, :, :]
        
        gate = lower_bound * torch.sigmoid(A_exp[None, None, :, None] * gate_input) * RCP_LN2
    else:
        gate = lower_bound * torch.sigmoid(g) * RCP_LN2
    
    # 4. Cumulative sum (chunk-wise for BT=16)
    NT = (T + BT - 1) // BT
    g_cumsum = torch.zeros(B, T, H, D, dtype=torch.float32, device=device)
    for i_chunk in range(NT):
        start = i_chunk * BT
        end = min(start + BT, T)
        g_cumsum[:, start:end] = torch.cumsum(gate[:, start:end], dim=1)
    
    # 5. Initialize state and output
    if initial_state is not None:
        h = initial_state.clone()
    else:
        if state_v_first:
            h = torch.zeros(B, H, D, D, dtype=torch.float32, device=device)  # [B, H, V=D, K=D]
        else:
            h = torch.zeros(B, H, D, D, dtype=torch.float32, device=device)  # [B, H, K=D, V=D]
    
    o = torch.zeros(B, T, H, D, dtype=dtype, device=device)
    
    # 6. Process chunk by chunk (BT=16)
    for i_chunk in range(NT):
        start = i_chunk * BT
        end = min(start + BT, T)
        chunk_len = end - start
        
        # Extract chunk
        q_c = q_norm[:, start:end].float()  # [B, chunk_len, H, D]
        k_c = k_norm[:, start:end].float()
        v_c = v[:, start:end].float()
        g_c = g_cumsum[:, start:end]  # [B, chunk_len, H, D]
        beta_c = beta_s[:, start:end]  # [B, chunk_len, H]
        
        # Process each (b, h) independently
        for b in range(B):
            for h_idx in range(H):
                q_bh = q_c[b, :, h_idx, :]  # [chunk_len, D]
                k_bh = k_c[b, :, h_idx, :]
                v_bh = v_c[b, :, h_idx, :]
                g_bh = g_c[b, :, h_idx, :]  # [chunk_len, D]
                beta_bh = beta_c[b, :, h_idx]  # [chunk_len]
                h_bh = h[b, h_idx]  # [V=D, K=D] state
                
                # Gate recentering
                mid = min(BT // 2, chunk_len - 1)
                g_mid = g_bh[mid]  # [D]
                g_recentered = g_bh - g_mid[None, :]  # [chunk_len, D]
                
                # Compute Aqk and Akk with recentered gates
                q_scaled = q_bh * torch.exp2(g_recentered)  # [chunk_len, D]
                k_scaled = k_bh * torch.exp2(-g_recentered)
                
                Aqk = torch.matmul(q_scaled, k_scaled.T) * scale  # [chunk_len, chunk_len]
                Aqk = torch.tril(Aqk)
                
                Akk = torch.matmul(k_bh * torch.exp2(g_recentered), k_scaled.T)  # [chunk_len, chunk_len]
                Akk = Akk * beta_bh[:, None]
                Akk = torch.tril(Akk, diagonal=-1)
                
                # NPU does not implement float64 tensors. Keep the higher
                # precision reference on CPU/CUDA, but use float32 on NPU.
                solve_dtype = torch.float32 if device.type == "npu" else torch.float64
                A_inv = torch.eye(chunk_len, dtype=solve_dtype, device=device)
                Akk_solve = Akk.to(solve_dtype)
                for i in range(1, chunk_len):
                    for j in range(i):
                        A_inv[i, :] -= Akk_solve[i, j] * A_inv[j, :]
                A_inv = A_inv.float()
                
                # Compute w and u
                k_weighted = k_bh * beta_bh[:, None] * torch.exp2(g_bh)  # [chunk_len, D]
                v_weighted = v_bh * beta_bh[:, None]  # [chunk_len, D]
                
                w = torch.matmul(A_inv, k_weighted)  # [chunk_len, D]
                u = torch.matmul(A_inv, v_weighted)  # [chunk_len, D]
                
                # Compute delta correction and output based on state layout
                g_last = g_bh[-1]  # [D]
                g_diff = g_last[None, :] - g_bh  # [chunk_len, D]
                k_gated = k_bh * torch.exp2(g_diff)  # [chunk_len, D]
                q_gated = q_bh * torch.exp2(g_bh)  # [chunk_len, D]
                
                if state_v_first:
                    # h is [V, K], kernel does: v_new = u - w @ h.T
                    v_new = u - torch.matmul(w, h_bh.T)  # [chunk_len, V]
                    
                    # Output: o = q_gated @ h.T * scale + Aqk @ v_new
                    o_bh = torch.matmul(q_gated, h_bh.T) * scale + torch.matmul(Aqk, v_new)
                    o[b, start:end, h_idx, :] = o_bh.to(dtype)
                    
                    # State decay: h = h * exp2(g_last)[None, :] (broadcast along K dimension)
                    h_bh = h_bh * torch.exp2(g_last)[None, :]  # [V, K]
                    
                    # State update: h += v_new.T @ k_gated
                    delta_state = torch.matmul(v_new.T, k_gated)  # [V, K]
                    h_bh = h_bh + delta_state
                    
                else:
                    # h is [K, V], kernel does: v_new = u - w @ h
                    v_new = u - torch.matmul(w, h_bh)  # [chunk_len, V]
                    
                    # Output: o = q_gated @ h * scale + Aqk @ v_new
                    o_bh = torch.matmul(q_gated, h_bh) * scale + torch.matmul(Aqk, v_new)
                    o[b, start:end, h_idx, :] = o_bh.to(dtype)
                    
                    # State decay: h = h * exp2(g_last)[:, None] (broadcast along V dimension)
                    h_bh = h_bh * torch.exp2(g_last)[:, None]  # [K, V]
                    
                    # State update: h += k_gated.T @ v_new
                    delta_state = torch.matmul(k_gated.T, v_new)  # [K, V]
                    h_bh = h_bh + delta_state
                
                h[b, h_idx] = h_bh
    
    return o, h


@pytest.mark.parametrize("B,T,H,D", [
    (1, 16, 1, 128),   # Single chunk
    (1, 17, 1, 128),   # Partial chunk
    (1, 32, 2, 128),   # Two full chunks
    (1, 33, 2, 128),   # Two chunks + partial
])
@pytest.mark.parametrize("device", ["npu", "cuda"])
def test_kda_bt16_vs_torch_reference(B, T, H, D, device):
    """Test KDA BT=16 against exact PyTorch reference (no FLA dependency)."""
    if device == "npu":
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            pytest.skip("NPU not available")
        torch.npu.set_device(0)
    elif device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.cuda.set_device(0)
    
    torch.manual_seed(42)
    
    # Generate fp32 inputs for reference
    q = torch.randn(B, T, H, D, dtype=torch.float32, device=device)
    k = torch.randn(B, T, H, D, dtype=torch.float32, device=device)
    v = torch.randn(B, T, H, D, dtype=torch.float32, device=device)
    g = torch.randn(B, T, H, D, dtype=torch.float32, device=device)
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device)
    A_log = torch.randn(H, dtype=torch.float32, device=device)
    dt_bias = torch.randn(H * D, dtype=torch.float32, device=device) * 0.1
    
    scale = D ** -0.5
    lower_bound = -5.0
    
    # Reference: fp32
    o_ref, _ = torch_reference_kda_bt16(
        q, k, v, g, beta,
        scale=scale,
        lower_bound=lower_bound,
        A_log=A_log,
        dt_bias=dt_bias
    )
    
    # Kernel: bf16
    q_bf16 = q.to(torch.bfloat16).contiguous()
    k_bf16 = k.to(torch.bfloat16).contiguous()
    v_bf16 = v.to(torch.bfloat16).contiguous()
    
    o_proto, _ = kda_bt16_fwd(
        q_bf16, k_bf16, v_bf16, g, beta,
        scale=scale,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=lower_bound,
        A_log=A_log,
        dt_bias=dt_bias,
        output_final_state=False,
    )
    
    # Compare
    max_diff = (o_proto.float() - o_ref).abs().max().item()
    relative_err = max_diff / (o_ref.abs().max().item() + 1e-12)
    
    print(f"\n[B={B}, T={T}, H={H}, D={D}, device={device}]")
    print(f"  max_diff: {max_diff:.6e}")
    print(f"  relative_err: {relative_err:.6e}")
    print(f"  o_ref range: [{o_ref.min().item():.3f}, {o_ref.max().item():.3f}]")
    print(f"  o_proto range: [{o_proto.min().item():.3f}, {o_proto.max().item():.3f}]")
    
    # Assertions
    assert torch.isfinite(o_proto).all(), "Kernel output has NaN/Inf"
    assert torch.isfinite(o_ref).all(), "Reference output has NaN/Inf"
    
    # Correctness check with realistic tolerance for bf16
    assert max_diff < 1e-2, f"max_diff {max_diff:.3e} exceeds threshold 1e-2"
    assert relative_err < 0.05, f"relative_err {relative_err:.3e} exceeds threshold 0.05"


@pytest.mark.parametrize("device", ["npu", "cuda"])
def test_kda_bt16_simple_gate(device):
    """Test without A_log/dt_bias (simple gate transformation)."""
    if device == "npu":
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            pytest.skip("NPU not available")
    elif device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    
    torch.manual_seed(123)
    B, T, H, D = 1, 32, 2, 128
    
    q = torch.randn(B, T, H, D, dtype=torch.float32, device=device)
    k = torch.randn(B, T, H, D, dtype=torch.float32, device=device)
    v = torch.randn(B, T, H, D, dtype=torch.float32, device=device)
    g = torch.randn(B, T, H, D, dtype=torch.float32, device=device)
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device)
    
    scale = D ** -0.5
    
    # A_log = zeros means exp(A_log) = 1, so gate = lower_bound * sigmoid(g) * RCP_LN2
    A_log = torch.zeros(H, dtype=torch.float32, device=device)
    
    # Reference
    o_ref, _ = torch_reference_kda_bt16(
        q, k, v, g, beta,
        scale=scale,
        lower_bound=-5.0,
        A_log=A_log,
        dt_bias=None
    )
    
    # Kernel
    q_bf16 = q.to(torch.bfloat16).contiguous()
    k_bf16 = k.to(torch.bfloat16).contiguous()
    v_bf16 = v.to(torch.bfloat16).contiguous()
    
    o_proto, _ = kda_bt16_fwd(
        q_bf16, k_bf16, v_bf16, g, beta,
        scale=scale,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=-5.0,
        A_log=A_log,
        dt_bias=None,
        output_final_state=False,
    )
    
    max_diff = (o_proto.float() - o_ref).abs().max().item()
    relative_err = max_diff / (o_ref.abs().max().item() + 1e-12)
    
    print(f"\n[Simple gate (no A_log/dt_bias), device={device}]")
    print(f"  max_diff: {max_diff:.6e}")
    print(f"  relative_err: {relative_err:.6e}")
    
    assert max_diff < 1e-2, f"max_diff {max_diff:.3e} exceeds threshold"
    assert relative_err < 0.05, f"relative_err {relative_err:.3e} exceeds threshold"


@pytest.mark.parametrize("device", ["npu", "cuda"])
def test_kda_bt16_input_validation(device):
    """Test that input validation catches common errors."""
    if device == "npu":
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            pytest.skip("NPU not available")
    elif device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    
    B, T, H, D = 1, 32, 2, 128
    
    q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device).contiguous()
    k = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device).contiguous()
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device).contiguous()
    g = torch.randn(B, T, H, D, dtype=torch.float32, device=device)
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device)
    
    # Test 1: Non-contiguous q
    q_nc = q.transpose(1, 2)
    with pytest.raises(ValueError, match="q must be contiguous"):
        kda_bt16_fwd(q_nc, k, v, g, beta, scale=0.0883)
    
    # Test 2: Wrong dtype
    k_fp32 = k.float()
    with pytest.raises(TypeError, match="k must be bfloat16"):
        kda_bt16_fwd(q, k_fp32, v, g, beta, scale=0.0883)
    
    # Test 3: Shape mismatch (V != 128)
    v_wrong = torch.randn(B, T, H, 64, dtype=torch.bfloat16, device=device).contiguous()
    with pytest.raises(ValueError, match="prototype targets K=V=128"):
        kda_bt16_fwd(q, k, v_wrong, g, beta, scale=0.0883)
    
    print(f"\n[Input validation tests passed on {device}]")


@pytest.mark.parametrize("T", [16, 17])
@pytest.mark.parametrize("state_v_first", [True, False])
@pytest.mark.parametrize("device", ["npu", "cuda"])
def test_kda_bt16_with_initial_state(T, state_v_first, device):
    """Test with non-zero initial state to catch transpose bugs.
    
    T=16: Single chunk - tests state read, w@h, q@h, state update
    T=17: Multi-chunk - tests state propagation across chunks + partial chunk
    state_v_first=True/False: Tests both state layouts (critical for K=V=128)
    """
    if device == "npu":
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            pytest.skip("NPU not available")
    elif device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    
    torch.manual_seed(456 + T)
    B, H, D = 1, 1, 128
    
    q = torch.randn(B, T, H, D, dtype=torch.float32, device=device)
    k = torch.randn(B, T, H, D, dtype=torch.float32, device=device)
    v = torch.randn(B, T, H, D, dtype=torch.float32, device=device)
    g = torch.randn(B, T, H, D, dtype=torch.float32, device=device)
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device)
    A_log = torch.randn(H, dtype=torch.float32, device=device)
    dt_bias = torch.randn(H * D, dtype=torch.float32, device=device) * 0.1
    
    # Random initial state - MUST be fp32 for kernel
    initial_state = torch.randn(B, H, D, D, dtype=torch.float32, device=device)
    
    scale = D ** -0.5
    lower_bound = -5.0
    
    # Reference with initial state
    o_ref, ht_ref = torch_reference_kda_bt16(
        q, k, v, g, beta,
        scale=scale,
        lower_bound=lower_bound,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        state_v_first=state_v_first
    )
    
    # Kernel with initial state (must be fp32 + contiguous)
    q_bf16 = q.to(torch.bfloat16).contiguous()
    k_bf16 = k.to(torch.bfloat16).contiguous()
    v_bf16 = v.to(torch.bfloat16).contiguous()
    
    o_proto, ht_proto = kda_bt16_fwd(
        q_bf16, k_bf16, v_bf16, g, beta,
        scale=scale,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=lower_bound,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state.contiguous(),  # FP32, not bf16!
        output_final_state=True,
    )
    
    # Compare outputs
    max_diff_o = (o_proto.float() - o_ref).abs().max().item()
    relative_err_o = max_diff_o / (o_ref.abs().max().item() + 1e-12)
    
    # Compare final states
    max_diff_h = (ht_proto.float() - ht_ref).abs().max().item()
    relative_err_h = max_diff_h / (ht_ref.abs().max().item() + 1e-12)
    
    print(f"\n[initial_state: T={T}, state_v_first={state_v_first}, device={device}]")
    print(f"  Output: max_diff={max_diff_o:.6e}, relative_err={relative_err_o:.6e}")
    print(f"  State:  max_diff={max_diff_h:.6e}, relative_err={relative_err_h:.6e}")
    
    # Critical assertions for transpose bugs
    assert torch.isfinite(o_proto).all(), "Kernel output has NaN/Inf"
    assert torch.isfinite(o_ref).all(), "Reference output has NaN/Inf"
    assert torch.isfinite(ht_proto).all(), "Kernel final_state has NaN/Inf"
    assert torch.isfinite(ht_ref).all(), "Reference final_state has NaN/Inf"
    
    assert max_diff_o < 1e-2, f"output max_diff {max_diff_o:.3e} exceeds threshold 1e-2"
    assert relative_err_o < 0.05, f"output relative_err {relative_err_o:.3e} exceeds threshold 0.05"
    
    assert max_diff_h < 1e-2, f"state max_diff {max_diff_h:.3e} exceeds threshold 1e-2"
    assert relative_err_h < 0.05, f"state relative_err {relative_err_h:.3e} exceeds threshold 0.05"
