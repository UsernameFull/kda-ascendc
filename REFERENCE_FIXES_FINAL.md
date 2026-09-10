# KDA BT=16 Reference Implementation Final Fixes

**Date**: 2026-09-05  
**Status**: All P0/P1/P2 issues resolved - Ready for validation testing

---

## Executive Summary

This document records the **complete rewrite** of the PyTorch reference implementation and associated fixes to validation/testing infrastructure.

**Previous Status**: Reference implementation contained fundamental mathematical errors that made it unsuitable as a correctness oracle.

**Current Status**: Reference now implements the exact KDA BT=16 mathematical formulation matching the Triton kernel line-by-line.

---

## Critical Issues Fixed (P0)

### P0-1: L2 Normalization Error ✅

**Problem**: Reference multiplied normalized vectors by `sqrt(D)`, making `||q|| ≈ 11.31` instead of `||q|| = 1`.

```python
# ❌ Before
q_norm = F.normalize(q.float(), p=2, dim=-1) * (D ** 0.5)

# ✅ After (exact kernel match)
eps = 1e-6
q_norm = q.float() / torch.sqrt(torch.sum(q.float() ** 2, dim=-1, keepdim=True) + eps)
```

**Impact**: QK dot products differed by ~128× magnitude.

---

### P0-2: Missing Gate in Output Equation ✅

**Problem**: Output equation missed `q * exp2(g)` gating term.

```python
# ❌ Before
o_bh = torch.matmul(q_bh, h_bh) * scale + Aqk @ v_new

# ✅ After
q_gated = q_bh * torch.exp2(g_bh)
o_bh = torch.matmul(q_gated, h_bh) * scale + torch.matmul(Aqk, v_new)
```

**Impact**: Cross-chunk contributions to output were mathematically incorrect from 2nd chunk onward.

---

### P0-3: Missing Gate in State Update ✅

**Problem**: State recurrence missed `k * exp2(g_last - g)` decay term.

```python
# ❌ Before
h_bh = h_bh * decay[None, :]
h_bh = h_bh + torch.matmul(k_bh.T, v_new)

# ✅ After
h_bh = h_bh * torch.exp2(g_last)[:, None]  # [V=D, K=D]
g_diff = g_last[None, :] - g_bh
k_gated = k_bh * torch.exp2(g_diff)
delta_state = torch.matmul(v_new.T, k_gated)  # [D, D]
h_bh = h_bh + delta_state
```

**Impact**: State recurrence equation fundamentally different from kernel.

---

### P0-4: State Layout Transpose Bug ✅

**Problem**: Reference treated state as `[K, V]` while kernel uses `[V, K]` when `state_v_first=True`. Hidden by K=V=128.

```python
# ✅ After (correct [V, K] handling)
h = torch.zeros(B, H, D, D, ...)  # [B, H, V=D, K=D]
v_new = u - torch.matmul(w, h_bh)  # w @ h, not w @ h^T
q_gated = q_bh * torch.exp2(g_bh)
o_bh = torch.matmul(q_gated, h_bh) * scale  # q @ h, not q @ h^T
delta_state = torch.matmul(v_new.T, k_gated)  # [D, D]
```

**Impact**: Would cause silent errors if K ≠ V in future.

---

### P0-5: Tests Missing A_log Parameter ✅

**Problem**: Tests called kernel with `use_gate_in_kernel=True` but didn't provide required `A_log`.

```python
# ❌ Before
o_proto, _ = kda_bt16_fwd(
    ...,
    use_gate_in_kernel=True,
    # Missing A_log - causes ValueError before kernel launch
)

# ✅ After
A_log = torch.randn(H, dtype=torch.float32, device=device)
dt_bias = torch.randn(H * D, dtype=torch.float32, device=device) * 0.1
o_proto, _ = kda_bt16_fwd(
    ...,
    use_gate_in_kernel=True,
    A_log=A_log,
    dt_bias=dt_bias,
)
```

**Impact**: All 8 test cases would fail with ValueError immediately.

---

### P0-6 & P0-7: pytest.raises Match Errors ✅

**Problem**: Test expected error messages didn't match actual validation messages.

```python
# ❌ Before
with pytest.raises(TypeError, match="k and v must match q dtype"):

# ✅ After
with pytest.raises(TypeError, match="k must be bfloat16"):

# ❌ Before
with pytest.raises(ValueError, match="v.shape"):

# ✅ After
with pytest.raises(ValueError, match="prototype targets K=V=128"):
```

**Impact**: Validation tests would fail even when validation logic was correct.

---

## High-Priority Validation Fixes (P1)

### P1-8: Improved Shape Validation ✅

**Before**: Shape checks extracted dimensions from input then compared input to itself (tautology).

```python
# ❌ Before
HV, V = v.shape[2], v.shape[-1]
if v.shape != (B, T, HV, V):  # Always true!
```

**After**: Now handled by ndim check + prototype constraint `V=128`.

---

### P1-9: Added ndim and Empty Shape Checks ✅

**Problem**: Missing basic input validation caused confusing Python errors.

```python
# ✅ Added at start of _validate_kda_inputs()
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

B, T, H, K = q.shape
HV, V = v.shape[2], v.shape[-1]

if B <= 0 or T <= 0 or H <= 0 or K <= 0:
    raise ValueError(f"q dimensions must be positive, got B={B}, T={T}, H={H}, K={K}")
if HV <= 0 or V <= 0:
    raise ValueError(f"v dimensions must be positive, got HV={HV}, V={V}")
```

**Impact**: Now catches dimension errors before confusing unpack failures.

---

## Performance Benchmark Fixes (P2)

### P2-11: proto_breakdown STORE_A_INV=False ✅

**Problem**: Breakdown benchmark used `STORE_A_INV=True`, measuring debug path instead of production.

```python
# ❌ Before
A_inv = torch.empty(B, T, HV, BT, ...)
kda_bt16_kernel_k1[...](
    ..., A_inv, ...,
    STORE_A_INV=True,  # Extra GM write not in production
)

# ✅ After
# A_inv not allocated
kda_bt16_kernel_k1[...](
    ..., None, ...,
    STORE_A_INV=False,  # Production path
)
```

**Impact**: K1 timing now reflects actual production performance.

---

## Test Infrastructure

### Test Count Correction

**Expected test count**: 12 (not 26)

```
test_torch_reference.py:
  - test_kda_bt16_vs_torch_reference: 4 shapes × 2 devices = 8
  - test_kda_bt16_simple_gate: 1 × 2 devices = 2  
  - test_kda_bt16_input_validation: 1 × 2 devices = 2
  Total: 12

test_kda_bt16.py:
  - (existing FLA tests): 11

Grand total: 23 collected (not 26)
```

### Installation Instructions

**Corrected** (from docs):

```bash
# NPU environment
pip install -e ".[npu,test]"

# CUDA environment  
pip install -e ".[cuda,test]"
```

Not just `".[test]"` which would miss Triton backend.

---

## Reference Implementation Quality

### Verification Method

Implemented line-by-line match to kernel operations:

1. **L2Norm**: Exact eps=1e-6, no extra scaling
2. **Gate cumsum**: Chunk-wise with RCP_LN2 scaling
3. **Gate recentering**: `g - g_mid` for Aqk/Akk
4. **Aqk/Akk**: `exp2(g_recentered)` scaling
5. **A inverse**: Forward substitution in fp64
6. **w/u**: `A_inv @ (k*beta*exp2(g))`, `A_inv @ (v*beta)`
7. **v_new**: `u - w @ h` (correct [V,K] layout)
8. **Output**: `(q * exp2(g)) @ h * scale + Aqk @ v_new`
9. **State**: `h * exp2(g_last) + (k * exp2(g_last-g))^T @ v_new)^T`

### Expected Tolerance

```python
assert max_diff < 1e-2, "Absolute error threshold"
assert relative_err < 0.05, "Relative error threshold (5%)"
```

Realistic for bf16 kernel vs fp32 reference.

---

## Breaking Changes

None - these are fixes to unreleased test infrastructure.

---

## Files Modified

| File | Changes | Lines |
|------|---------|-------|
| `tests/test_torch_reference.py` | Complete rewrite of reference | 330 |
| `src/kda_bt16/kernels.py` | Added ndim/empty checks | +28 |
| `benchmarks/bench_bt16.py` | STORE_A_INV=False | ~5 |

---

## Next Steps

1. **Run actual tests on NPU/CUDA hardware**:
   ```bash
   cd /workspace/kda/kda_bt16
   pytest tests/test_torch_reference.py -v
   # Expected: 12 passed (8 + 2 + 2)
   ```

2. **Verify correctness with complete suite**:
   ```bash
   pip install -e ".[npu,test]"
   pytest tests/ -v
   # Expected: 23 collected (12 + 11)
   ```

3. **Run benchmark**:
   ```bash
   python benchmarks/bench_bt16.py --device npu
   # Should now show fair BF16 vs BF16 comparison
   ```

4. **If tests pass**: Update status from "validation WIP" to "validated"

5. **If tests fail**: Reference still has issues - do NOT use as oracle

---

## Risk Assessment

**Before this fix**:
- ❌ Reference mathematically incorrect (4 major errors)
- ❌ Tests would fail immediately (missing A_log)
- ❌ Validation tests would fail (wrong error messages)
- ❌ Benchmark measured debug path, not production
- 🔴 **Status: Validation system completely unreliable**

**After this fix**:
- ✅ Reference implements exact kernel math
- ✅ Tests have all required parameters
- ✅ Validation tests match actual errors
- ✅ Benchmark measures production path
- ✅ Proper ndim/shape checks
- 🟡 **Status: Ready for hardware validation testing**

**Remaining uncertainty**:
- No hardware test results yet (environment has no NPU/CUDA)
- Reference assumes single precision will match bf16 within 5% relative error
- If tests fail, may need tolerance tuning or deeper debugging

---

## Validation Checklist

Before declaring "validated":

- [ ] `pytest tests/test_torch_reference.py -v` shows 12 passed
- [ ] `pytest tests/test_kda_bt16.py -v` shows 11 passed (requires FLA)
- [ ] All tests use device="npu" or device="cuda" (no CPU kernel launch)
- [ ] Benchmark shows max_diff < 1e-2, relative_err < 0.05 vs fp32 gold
- [ ] K1 breakdown timing reflects production (STORE_A_INV=False)
- [ ] README/docs updated with correct test count and install instructions

---

## Conclusion

This round of fixes addressed **fundamental mathematical correctness** of the reference implementation, not just engineering polish.

The kernel itself appears correct based on code review - the validation system was the problem.

**Current classification**: 
- Kernel: Production candidate
- Validation: **Fixed but untested on hardware**
- Overall: **Validation WIP** → needs hardware test confirmation

Once tests pass on actual hardware, status can move to "Validated and ready for performance optimization".
