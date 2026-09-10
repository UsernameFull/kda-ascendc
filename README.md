# kda-bt16: Two-Stage Triton Kernels for Kimi Delta Attention (KDA) Forward

A compact, two-stage Triton implementation of the KDA (Kimi Delta Attention)
forward pass with `BT=16` chunks, written and tuned for Ascend NPU (910B) but
portable to any Triton backend (CUDA included). It reproduces the numerics of
FLA's production `chunk_kda` path while fusing the pipeline into two kernels.

> Research prototype: forward only, fixed-length sequences, bf16, `K=V=128`,
> MHA (`HV == H`). See [Limitations](#limitations).

## Why two stages at BT=16

The production FLA path serializes 9+ kernels per forward (gate pre-pass,
intra-chunk QK^T / forward-substitution solve / W-U recompute, then a separate
state-propagation kernel), with every intermediate dumped to HBM.

With `BT=16`:

- the intra-chunk solve `A = (I + Akk_lo)^{-1}` is a **16x16 forward
  substitution that fits inside a single kernel** — no inter-kernel
  communication is needed (`NC == 1`), and 16 is the Cube matrix-unit size on
  Ascend;
- the gate/chunk-cumsum, QK^T, solve, and W/U recompute collapse into **K1**
  (one program per `(batch, head, chunk)`);
- the delta-rule state recurrence and output collapse into **K2** with the
  state **resident on-chip** across the whole sequence (no per-chunk state
  round-trips), following the FlashKDA idea.

### Math (mirrors `fla/ops/kda/chunk_fwd.py`)

```
g      = cumsum(lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))) * RCP_LN2   # log2 domain
Aqk    = scale * masked_lower(q*exp2(g-mid) @ (k*exp2(mid-g))^T)                  # fp32
A      = (I + Akk_lo)^-1,   Akk_lo = masked_strict_lower(kk^T * beta * exp2(g_s - g_t))
w      = A @ (k * beta * exp2(g)),    u = A @ (v * beta)                          # K1 output
v_new  = u - w @ h                       # h: pre-chunk state
o      = scale * (q*exp2(g)) @ h + Aqk @ v_new
h      = h * exp2(g_last) + (k*exp2(g_last - g))^T @ v_new                        # K2
```

Numerics follow the production path exactly: `Aqk`/`Akk` and the 16x16 solve
are fp32 (`allow_tf32=False`); the K2 state/output dots are bf16 with fp32
accumulation, matching `chunk_gated_delta_rule_fwd_h` / `chunk_gla_fwd_o_gk`.
The l2-norm of q/k and the beta sigmoid are fused into both kernels (in-kernel
`rsqrt`, `eps=1e-6`).

## Results (Ascend 910B3, bf16, min latency across reps)

| shape | proto (BT=16) | FLA chunk64 | FLA chunk32 | speedup vs chunk64 |
|---|---|---|---|---|
| B=2 T=1024 H=4 | **1.19 ms** | 3.85 ms | 3.88 ms | 3.2x |
| B=2 T=4096 H=8 | **5.69 ms** | 9.31 ms | — | 1.6x |
| B=1 T=8192 H=32 | **20.5 ms** | 28.4 ms | — | 1.4x |

Precision (max-abs-diff vs fp32 recurrent gold, B=2 T=1024): `o` and `ht` diffs
are **bit-identical to FLA chunk_kda@64** (`o` 4.9e-04, `ht` 1.8e-03), and the
proto-vs-chunk64 cross-diff is 1.2e-04.

Implementation notes from tuning on 910B:

- K2 keeps `BV = V = 128` (full state resident). V-splitting doubles the
  duplicated `w/q/k/g` loads per head and measured ~2x slower despite more
  programs.
- `tl.range(num_stages=2/4)` gave no benefit; `NUM_STAGES=1` is fixed.
- Intermediates use `torch.empty` (never `torch.zeros`).
- K2's state/output dots were moved from fp32 to bf16 operand dots, which cut
  the K2 kernel alone from ~11.8 ms to ~10.0 ms at T=8192 with **no** precision
  change (it matches the official kernels' numerics).

Negative results (measured on 910B3, kept here so nobody re-chases them):

- **Single-kernel full fusion** (`use_fused_kernel=True`, K1 math recomputed
  per chunk inside the state loop, zero intermediate HBM traffic): ~1.5x
  *slower* than the two-stage split at large T (30.1 ms vs 20.5 ms at
  T=8192 H=32). Fusing collapses K1's `NT*BH`-program grid into `B*HV`
  serial programs; the lost grid parallelism costs far more than the removed
  inter-kernel barrier and round-trips. Kept as an experimental reference.
- **Register-only solves** in K1 (avoiding the 16x16 fp32 `Akkd`
  store/load/barrier): both forward substitution via `where+sum` row
  extraction and Neumann-doubling `(I+M)(I+M^2)(I+M^4)(I+M^8)` dots measured
  ~10% *slower* overall (K1 10.6 -> 11.8 ms at T=8192). The tiny global
  round-trip hits L2 and pipelines better than register shuffles on this
  backend.
- **num_warps** sweep {2,4,8} across K1 and K2: <0.6% spread. The Ascend
  JIT compiler already picks optimal occupancy for these fixed shapes.
- **allow_tf32=True** on Ascend NPU: not supported — `tl.dot` raises
  `AssertionError`. All matmuls use bf16 operands with fp32 accumulation.
- **tl.trans reuse**: precomputing `tl.trans(b_hc)` and `tl.trans(b_kg)` once
  per chunk and reusing for both v_new, output, and state dots: no measurable
  effect (<0.3%). `tl.trans` is a zero-cost register view on Ascend.
- **Cross-batch stream pipelining**: splitting K2 into per-batch launches on
  separate NPU streams to overlap K2(b=0) || K2(b=1): 55% *slower* at B=2
  T=1024. Grid halves from B*HV to HV programs per launch, and stream
  management overhead dominates. The default single-launch path with B*HV*NV
  programs is faster.
- **K2 manual prefetch** (load chunk i+1 data at start of iteration i to
  overlap DMA with matmul of chunk i): not measurable via `num_stages=2/4`
  which showed <0.1% difference. K2's data loads (~14KB/chunk: q/k/w/u/g/A)
  complete well before the matmuls finish; the state dependency chain
  (`b_h` depends on previous iteration) prevents useful overlap. Triton
  compiler on Ascend already schedules loads optimally.
- **Dynamic BV dispatch** (BV=64/32 for small B×H to increase program count):
  only BH=4 benefits (BV=64 is 4.7% faster than BV=128 at B=1 T=4096 H=4).
  For BH≥8, BV=128 is always fastest: the duplicated q/k/w/g loads from
  V-splitting dominate the occupancy gain. The crossover point is too narrow
  to justify host-side dispatch logic.
- **q_rsqrt/k_rsqrt scalar caching** (store per-token-head L2Norm reciprocal
  in K1, load in K2 instead of recomputing): K1 L2Norm costs 29-35% of K1
  time but K2 L2Norm costs only 0.5-3% of K2. Net savings from caching
  rsqrt scalars ≈ K2 norm reduction ≈ 150us — negligible. The expensive
  norm is in K1 (which already computes and stores normalized q/k), and
  K2's reduction is nearly free because it operates on already-warm data
  in registers.
- **Aqk recompute in K2** (skip storing Aqk from K1, recompute q@k^T in K2
  to save GM traffic): Aqk is only 8 MiB out of a 288 MiB workspace
  (2.8%) and 0.01% of total K2 GM traffic. Even complete elimination
  is negligible. The actual workspace hog is g_cum (FP32, 44% of
  workspace).

## Profile (Ascend 910B3)

### K2 utilization

| metric | K2 achieved | peak | utilization |
|---|---|---|---|
| compute | ~2.7 TFLOPS | ~320 TFLOPS | ~0.8% |
| GM bandwidth | ~48 GB/s | ~1600 GB/s | ~3.0% |

Both utilization metrics are very low, confirming K2 is **neither compute-
nor bandwidth-bound**.

### Scaling analysis

**NT scaling** (BH=32 fixed, T varies) — K2 latency is linear in NT:

| T | NT | K2 (us) | K2/NT (us/chunk) |
|---|---|---|---|
| 1024 | 64 | 1525 | 23.8 |
| 2048 | 128 | 2733 | 21.4 |
| 4096 | 256 | 5157 | 20.1 |
| 8192 | 512 | 10037 | 19.6 |
| 16384 | 1024 | 19843 | 19.4 |

Linear regression: R²=1.0000. Each chunk adds a constant to the critical
path — the serial dependency chain is confirmed.

**BH scaling** (T=8192 fixed) — two regimes:

| BH | K2 total (us) | per-head (us) | per-chunk (us) | regime |
|---|---|---|---|---|
| 4 | 3796 | 949 | 1.85 | occupancy-limited |
| 8 | 4110 | 514 | 1.00 | occupancy-limited |
| 16 | 5125 | 320 | 0.62 | **saturation** |
| 32 | 10175 | 318 | 0.62 | dependency-bound |
| 64 | 21483 | 336 | 0.66 | dependency-bound |

For BH≥16, per-head latency saturates at ~0.62 us/chunk — adding more
programs no longer helps. The bottleneck transitions from device occupancy
to intra-head serial dependency.

**Conclusion**: K2 is latency/dependency-bound. The critical path is
512 sequential state updates per head. Hardware resources (99% of Cube
and BW capacity) idle waiting for each chunk's state to complete.

**Micro-level profiling** (single head, BT=16, K=V=128):

- Per-step time: **19.2 us** (measured via single-step timing)
- Per-step GM traffic: **28.5 KB** (25 KB reads + 4 KB write)
- Effective L2 BW: **1465 GB/s** (92% of 1600 GB/s peak)
- Matmul compute: 1.64 MFLOP → 5 ns at 320 TFLOPS (negligible)
- **Conclusion**: K2 is memory-bandwidth-bound at the per-step level.
  The macro-level "latency-bound" behavior comes from 512 serial steps,
  each of which is BW-saturated.

### K2 bytes/step breakdown (BT=16, K=V=128, BV=128, STATE_V_FIRST)

```
Tensor        Shape       Dtype   Bytes    GM     Optimization
──────────────────────────────────────────────────────────────
q             16×128      bf16    4,096    read   precompute qg → eliminate
k             16×128      bf16    4,096    read   precompute k_gated → eliminate
g_cumsum      16×128      fp32    8,192    read   BF16 → save 50%
g_last        128         fp32      512    read   extract from b_g → free
w             16×128      bf16    4,096    read   K1→K2 fusion → eliminate
u             16×128      bf16    4,096    read   K1→K2 fusion → eliminate
Aqk           16×16       bf16      512    read   too small → ignore
──────────────────────────────────────────────────────────────
Subtotal reads                         25,088 B (24.5 KB)
o (write)     16×128      bf16    4,096    write  mandatory
──────────────────────────────────────────────────────────────
TOTAL per step                         29,184 B = 28.5 KB
```

State h: 128×128 fp32 = 64 KB, resident in UB, load-once + store-once
(confirmed: per_step constant at 39.6 us across T=1K~8K).

### Negative results (round 3 — L2 traffic reduction)

- **g_cum FP32→BF16**: rel_err=3.3e-4 (acceptable). K2 timing unchanged
  (20.22 vs 20.1 ms). At 95% L2 BW utilization, halving g_cum's dtype
  doesn't reduce actual cache line fetches. Reverted.
- **g_last extraction from b_g**: 16×128 comparison + sum pattern is
  slower than 512 B direct GM load (20.49 vs 20.1 ms). The register
  comparison overhead exceeds the bandwidth savings. Reverted.
- **Both combined**: 3.6% regression (20.69 ms). The bf16→fp32 conversion
  + comparison overhead exceeds the 4.5 KB/step savings.

### Round 4 — load ablation: the bandwidth theory was WRONG

Load-ablation K2 variants (skip loading tensor X, replace with register
zeros, keep ALL compute instructions; timing delta = X's true critical-path
cost, i.e. the hard ceiling of any fusion of X):

| removed loads | bytes saved | time saved | fusion ceiling |
|---|---|---|---|
| w+u (M1 target) | 8 KB | **0.5%** | **1.00×** |
| w+u+q+k | 16 KB | 2.6% | 1.03× |
| all incl. g_cum | 24.5 KB | 23.3% | 1.30× |

**Producer-consumer fusion is dead as a latency optimization**: w/u load
latency hides entirely under the b_h dependency chain. The earlier
"92% L2 BW" was bytes/time correlation, not causation.

Floor decomposition (19.1 us/step):

| component | us/step | share |
|---|---|---|
| loop floor (loads+dots+exp all off) | 0.3 | 1.5% |
| exp2 layer | ~0 | 0% |
| **dots + l2norm layer** | **14.6** | **76.5%** |
| exposed load latency | 4.2 | 22% |

### Round 4 — dot-call probes: tl.dot runs on VECTOR units

- Marginal cost per additional 16×128×128 dot: **~4.7 us/call**, flat in N.
- Single-dot scaling: T(M) ≈ 5.6 us + 78 ns/row (M=16→128, K=N=128).
  Even the *marginal* regime runs at ~0.4 TFLOPS effective — two orders
  below the Cube; triton_ascend lowers these small dots to vector-FMA
  loops with heavy per-call setup.
- Consequence: an M=128 dot costs only ~2.6× an M=16 dot while doing 8×
  the work → **big batched dots amortize the fixed cost**; small serial
  dots are worst-case.
- Attempted 4→3 dot merge (row-concat [qg;w] via tl.join/permute/reshape):
  **MLIRCompilationError — BiShengHIR UB overflow** (requests 2.3 MB >
  192 KB UB), both permute syntax variants. Not implementable in this
  Triton version.

### Revised performance model

    T_K2/step ≈ 0.3 (loop) + 14.6 (4 vector-path dots) + 4.2 (exposed loads)

Levers ranked by measured headroom:
1. **Move the 4 GEMMs to the Cube unit** (Ascend C MMA): dot layer
   14.6 → <2 us plausible. Whole-op estimate ~1.4–1.6×.
2. **Batch small dots into large ones** (blockwise/two-pass): probe shows
   large dots amortize fixed cost even on the vector path.
3. Exposed-load removal via fusion: ≤1.28× hard ceiling, real ~0.5%. Dead.

### Round 5 — Cube path calibration: GO signal

Triton compile flags (`enable_mixed_cv`, `mix_mode="aic"`,
`tile_mix_cube_loop`) all produce bit-identical binaries and timing — no
free lunch inside Triton.

Native-path reference via torch.mm (aclnn optimized kernels) measured with
NPU-graph replay (100 chained ops, host overhead removed):

| shape | chained latency | effective |
|---|---|---|
| [16,128]@[128,128] | **1.64 us** | 319 GFLOPS |
| [16,16]@[16,128] | 1.31 us | 50 GFLOPS |
| [128,16]@[16,128] | 1.80 us | 291 GFLOPS |
| [512,512]@[512,512] | 5.81 us | 46 TFLOPS |
| **K2-mimic 4-dot chain w/ deps** | **5.82 us/step** | **2.51× vs Triton's 14.6 us dot layer** |

Projections:
- Separate-kernel bound (measured): K2 ≈ 19.1 − 14.6 + 5.8 = **10.3 us/step**
  → whole-op ≈ 20.3 → ~15.8 ms (**1.29×**).
- Fused Ascend C bound (one launch, UB/L1 residency, no GM roundtrip between
  dots): dot region plausibly 3–5 us → K2 ~7–9 us/step → **~1.5× whole-op**.
- Note: graph-replay numbers include inter-kernel device gaps (~0.5–1
  us/kernel); true single-kernel durations are lower — more upside for fusion.

### Round 6 — AscendC raw-launch pipeline verified

`aclab/` contains a working, minimal Ascend C development loop on this box:

    kernel.cpp → cmake(ascendc.cmake) → device_aiv.o
               → mskl launcher gen → pyACL launch → correct results

Verified: `y = 2*x` kernel compiled with official `ascendc_library` macro,
loaded via `mskl.get_kernel_from_binary`, launched from Python with integer
GM addresses (pyacl malloc/memcpy), numerically correct on device.

Gotchas solved (documented in aclab/README.md + setup_shims.sh):
- toolchain path shims for bisheng/ccec; writable ASCEND_HOME_PATH shim
- args must be integer device addrs; fabricated tiling context bypasses regbase
- bare TBuf across MTE→V→MTE silently drops compute → use TQue barriers

This unblocks M1 (`k2_step.cpp`, real K2 single step with Cube matmuls).
Level-B cube primitive references confirmed present locally in
/workspace/vllm-ascend/csrc/batch_matmul_transpose/op_kernel/.

### Workspace breakdown (B=1 T=8192 H=32: 288 MiB total)

| tensor | shape | size | % | note |
|---|---|---|---|---|
| g_cum | `[B,T,HV,K]` fp32 | 128 MiB | 44% | main compression target |
| w | `[B,T,HV,K]` bf16 | 64 MiB | 22% | |
| u | `[B,T,HV,V]` bf16 | 64 MiB | 22% | |
| Akkd | `[B,T,HV,BT]` fp32 | 16 MiB | 5.6% | solve scratch |
| Aqk | `[B,T,HV,BT]` bf16 | 8 MiB | 2.8% | |
| Akk | `[B,T,HV,BT]` bf16 | 8 MiB | 2.8% | written by K1, never read by K2 |

`o` (`[B,T,HV,V]` bf16, 64 MiB) is the op output, not counted above.

**Aqk recompute** (skip K1 store, recompute q@k^T in K2): not worthwhile.
Aqk is 2.8% of workspace and 0.01% of K2's total GM traffic. The real
target for workspace compression is `g_cum` (FP32 → BF16+chunk-scale
saves ~75%).

## Requirements

- Python >= 3.10, PyTorch >= 2.1
- A Triton backend: `triton` (CUDA) or `triton-ascend` (NPU, with `torch-npu` +
  CANN). The kernels themselves are plain Triton and backend-agnostic.
- Optional, for the correctness tests / benchmark gold references: a checkout
  of [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
  (main). `tests/` are skipped when it is unavailable.

## Install

```bash
pip install -e .            # package only (no NPU deps)
pip install -e ".[npu]"     # with triton-ascend / torch-npu
git clone https://github.com/fla-org/flash-linear-attention.git ../fla   # optional
```

## Usage

```python
import torch
from kda_bt16 import kda_bt16_fwd

q = torch.randn(B, T, H, 128, dtype=torch.bfloat16, device="npu")
k = torch.randn(B, T, H, 128, dtype=torch.bfloat16, device="npu")
v = torch.randn(B, T, H, 128, dtype=torch.bfloat16, device="npu")
g = torch.randn(B, T, H, 128, dtype=torch.float32, device="npu")   # raw gate
beta = torch.randn(B, T, H, dtype=torch.float32, device="npu")     # raw logits
A_log = torch.randn(H, dtype=torch.float32, device="npu")
dt_bias = torch.randn(H * 128, dtype=torch.float32, device="npu") * 0.1

o, ht = kda_bt16_fwd(
    q, k, v, g, beta,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    lower_bound=-5.0, A_log=A_log, dt_bias=dt_bias, state_v_first=True,
)
```

With `use_gate_in_kernel=False`, `g` must already be the chunk-local cumsum in
the log2 domain (e.g. produced by `kda_gate_chunk_cumsum`).

### Tests (requires FLA checkout; skipped otherwise)

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python -m pytest tests/ -x -q      # 11 tests
```

`tests/conftest.py` resolves the FLA root from `FLA_ROOT` (defaults to
`../fla` next to the repo).

### Benchmark + precision comparison

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python benchmarks/bench_bt16.py            # full
ASCEND_RT_VISIBLE_DEVICES=0 python benchmarks/bench_bt16.py --quick    # timing only
python benchmarks/bench_bt16.py --device cuda                           # on GPU
```

### Per-stage debug scripts

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python scripts/debug_k1.py   # K1 intermediates vs fp64 reference
ASCEND_RT_VISIBLE_DEVICES=0 python scripts/debug_k2.py   # K2 recurrence vs fp64 reference
```

## Repository layout

```
src/kda_bt16/
  kernels.py        # kda_bt16_kernel_k1 / _k2 / _fused + python driver
  __init__.py       # kda_bt16_fwd, kda_bt16_debug
tests/              # correctness vs fused_recurrent_kda (gold) + chunk_kda
benchmarks/         # latency + precision comparison vs FLA
scripts/            # per-stage numerical debug (uses kda_bt16_debug intermediates)
aclab/              # Ascend C raw-kernel lab: build/launch harness + K2 Cube
                    # kernels + the full experiment log (aclab/README.md)
docs/KDA_ANALYSIS.md  # deep-dive analysis of FLA's KDA backends that motivated this design
```

## Limitations

- Forward only (no backward), fixed-length sequences (no varlen / chunk
  continuation offset), bf16, `K=V=128`, `HV==H`, `state_v_first` layouts.
- Not yet integrated with FLA's dispatch registry (call `kda_bt16_fwd`
  directly).

## Compiler bug reports (Triton-Ascend 3.2.1 / BiSheng)

Two independent compiler bugs blocked the M2 persistent/block-persistent
K2. Both are reduced to minimal repros (AIC-only, 16×128×128 bf16 matmul,
correctness vs torch reference). Each is a clear upstream case.

### Bug 1: runtime-trip-count `for` loop + Cube matmul → wrong result

Minimal repro (`aclab/` build pipeline; kernel does one
`Mmad(16,128,128)` per iteration over `w[t] @ h^T`, Fixpipe out):

| structure | result |
|---|---|
| no loop (single d1) | d1 err 4.8e-7 (correct) |
| fully unrolled NT=2 (no `for`) | 4.8e-7 (correct) |
| `for (int t = 0; t < NT; t++)` (NT=1) | **3.58 (wrong)** |
| `for (int u = 0; u < 2; u++)` + `#pragma unroll` | 4.8e-7 (compiler unrolls) |
| `for (int u = 0; u < NT; u++)` + `#pragma unroll 4/1` | **wrong** (unroll ineffective on runtime count) |

Trigger: runtime-control-flow + `LoadData`/`Mmad`/`Fixpipe` pipeline.
Even a single iteration (NT=1) inside `for` is wrong, so it is a lowering
bug, not a loop-carried-state issue.

### Bug 2: multiple Mmad executions in one kernel → cross-block corruption

Same `Mmad(16,128,16)` × 8 sub-mmad (d4 = v_new^T @ kg, m-tiled) fully
unrolled (no `for`):

| structure | result |
|---|---|
| B=1 (single block) | d4 err 1.2e-7 (correct) |
| B=2 (two unrolled blocks) | **both blocks wrong** (2.09 / 1.75) |

The PRESENCE of a second Cube block corrupts the FIRST block. Contrast:
a B=8 kernel of SINGLE-mmad blocks (d1) is fully correct — the bug is
specific to multiple `Mmad`/`LoadData` executions (or multi-sub-mmad
blocks) sharing L0A/L0B/L0C across sequential unrolled blocks.

### Impact

- Single-cube-per-block workaround works: MIX1_blockB (AIC 8× d1 + AIV
  8× v_new/qg/kg/vnewT) B=8 fully correct.
- M2 persistent (512-step) and block-persistent with d2/d3/d4 (10 Mmad
  per block) are blocked until these are fixed upstream or a native
  CANN (ccec/opc) path is used.

## Roadmap

**Current status**: the two-stage Triton path (`kda_bt16_fwd`) is the only
shipping implementation. M1 (Ascend C two-MIX single-step K2) is verified
correct at the kernel level — all 10 outputs match the torch reference
(bf16-level), graph-pipelined throughput ~0.07 us/step — but it is **not
integrated**: the host-orchestrated wrapper was removed (see "Wrapper
removed" in `aclab/README.md`), and rounds 51–52 measured the
per-head-launch approach at ~35.5 ms projected for H=32 T=8192 vs 20.3 ms
for Triton. Integration needs multi-core scheduling, not another wrapper.
M2 persistent/block-persistent is **blocked by two Triton-Ascend compiler
bugs** (see "Compiler bug reports" above): dynamic-loop + Cube and
multi-Mmad block reuse. M2 resumes after an upstream fix or via a native
CANN (ccec/opc) kernel path.

Earlier analysis: K2 is NOT bandwidth-bound. 76.5% of each step is
`tl.dot` execution on vector units (~0.4 TFLOPS effective, ~3-5 us fixed
cost per call). Fusion and traffic compression are dead ends (≤1.28× hard
ceiling, measured ~0.5% real).

Remaining directions (priority order):

- **Compiler fix or native CANN path for persistent K2**: unblocks
  M2 (single 512-step kernel or block-persistent B=8). Two upstream bug
  reports ready (dynamic-loop+cube; multi-Mmad reuse). Native ccec/opc
  kernels may not share these bugs.
- **Blockwise recurrence / two-pass** (revived with new justification):
  transition matrices M_i = diag(exp2(g_last)) − kg_i^T w_i depend only on
  chunk inputs → computable for ALL chunks in parallel with large batched
  dots, which amortize the per-call overhead (M=128 dot costs 2.6× an
  M=16 dot for 8× the work). Then affine scan compose + parallel replay.
  NOTE: depends on the same compiler fixes (batched dot sequences).
- **Prefill/decode separation**: decode (T=1) uses a single fused recurrent
  kernel without BT=16 chunking, Aqk, or triangular solve. Required for
  production inference (vLLM/ROLL).
- **g_cum workspace compression** (FP32 → BF16): halves g_cum
  (128 MiB→64 MiB at B=1 T=8192 H=32, i.e. 288→224 MiB total). Zero
  latency effect (measured); do it for memory footprint only.
- varlen / `bc` continuation support, backward pass
- Wrapping as a FLA backend for drop-in `chunk_kda` use

Frozen (measured dead ends):
- producer-consumer W/U/q/k fusion (ceiling 1.00–1.03×)
- g_cum dtype change for latency (timing unchanged)
- g_last register extraction (+1.9% regression)
- prefix scan via D×D matmul levels (matmul-bound; revisit only after Cube
  path lands)
- Triton micro-tuning (num_warps/stages/BV/etc. all <5% or negative)
- **K1→K2 operand precompute** (K1 stores `qg`/`k_gated`/`exp2_g_last` so K2
  skips the exp2 and the raw q/k loads): implemented and measured, then
  removed. Correct, but only −1.6% end-to-end at B=1 T=8192 H=32 (−7.5% at
  B=2 T=1024 H=4, where K2 dominates), because K1 pays back most of what K2
  saves (K1 10.8→12.7 ms, K2 9.9→7.7 ms). Workspace grows 288→416 MiB even
  with `exp2_g_last` compacted to `[B,NT,HV,K]`. Consistent with the round-4
  load-ablation ceiling. Revisit only if K1 moves to Cube.

## License

MIT (see `LICENSE`). Reference implementations used only by `tests/` and
`benchmarks/` come from [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (MIT).