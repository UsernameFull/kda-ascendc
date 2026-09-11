# What the Triton KDA path actually compiles to

Date: 2026-09-11.  Device: NPU 1, shape `[1, 8192, 32, 128]`, `lower_bound=-1`.

Sources inspected: `src/kda_bt16/kernels.py` and the triton-ascend artifacts in
`/root/.triton/cache/<hash>/kda_bt16_kernel_{k1,k2}.{ttir,ttadapter,npubin}`.
The `.ttadapter` file is the generated linalg-level device code and is the
closest thing to an AscendC kernel that Triton produces.

## How the reference is built

`kda_bt16_kernel_k1` - grid `NT * BH` (16384 programs here), one program per
(batch, head, chunk).  It loads raw `q/k/v/g/beta`, runs the q/k l2 norm, the
gate cumsum, both Gram dots, the 16-row forward substitution and the `w`/`u`
dots.  Every `tl.dot` becomes `linalg.matmul {input_precison = "ieee"}` in the
adapter IR, i.e. a Cube MMAD.

`kda_bt16_kernel_k2` - grid `B * HV * NV` (64 programs, `Block Num 64`,
`Mix Block Num 128`), **one launch for the whole sequence**: `b_h` is an
`scf.for ... iter_args(%h: tensor<64x128xf32>)`, so the fp32 state stays in the
program across all 512 chunks.  Per chunk the adapter IR contains exactly

- `linalg.transpose` 64x128 bf16 -> 128x64: the state cast once, transposed
  once, and **shared by two matmuls** - `w @ h^T` for `v_new` and
  `q*exp2(g) @ h^T` for `o`,
- `linalg.matmul` 16x128x64 (that pair) and `linalg.matmul` 16x16x64
  (`A @ v_new`),
- `math.exp2` on the resident gate tiles (`exp2(g)`, `exp2(g_last - g)`),
- `linalg.transpose` 16x128 -> 128x16 (`k*exp2(g_last-g)`), `linalg.matmul`
  128x16x64, `linalg.transpose` 128x64 -> 64x128 (the `d4` tile back into the
  state layout), `arith.addf`.

The kernel is `mix_mode = "mix"` / `parallel_mode = "simd"`, so triton-ascend
inserts the AIC/AIV sync itself and the state never leaves the program.

## Measured split (msprof, one pass each)

| Triton kernel | blocks | ms |
|---|---:|---:|
| `kda_bt16_kernel_k1` | 16384 (mix 32768) | 11.090 |
| `kda_bt16_kernel_k2` | 64 (mix 128) | 18.793 |

Our `separated` path on the same device and shape, device time per kernel:

| stage | blocks | ms |
|---|---:|---:|
| preprocess | 16384 | 1.920 |
| gram | 16384 | 2.334 |
| solve_wu (AIV) | 16384 | 2.403 |
| solve_wu_cube | 16384 | 0.877 |
| kg_transpose | 16384 | 0.740 |
| k2_d12 / vnew / d34 / outstate | 64 blocks x 512 launches each | 3.082 / 2.956 / 3.385 / 3.781 |

K1: 7.53 ms ours vs 11.09 ms Triton (1.47x).  K2: 13.95 ms ours (including
`kg_transpose`) vs 18.79 ms Triton (1.35x).  Per chunk of the serial chain
Triton pays 18.793 / 512 = **36.7 us**; we pay 13.95 / 512 = **27.2 us**.

## What is worth borrowing

1. *Nothing structural.* The resident-state design is the whole point of their
   K2 and it is still 35% slower per chunk than our four-launch chain with GM
   round-trips.  With the `persistent_scan_cube` deadlock (see
   `ASCENDC_V1_KERNELS.md`) this closes the "one MIX kernel per sequence"
   line - do not restart it.
2. *Derive the second exponent from the resident tile.* We were re-reading
   `Gc` from GM in `k1_gram.cpp` (`ef` and `t0` both came from `Gc[x0]`).
   Removing the second 8 KB load per chunk measured 2.335 -> 2.334 ms, i.e.
   nothing: gram is issue/compute-bound, not load-bound, which agrees with the
   earlier no-load probe (2.23 ms).  Reverted.
3. *exp2 in the log2 domain.* We already store the gate in log2 units and we
   already re-centre on the chunk mid row, exactly like their `b_gm = b_g -
   b_gn`.  AscendC exposes `Exp`, `Ln`, `Reciprocal`, `Rsqrt` but no vector
   `Exp2`, so the per-element `* LN2` stays.
4. *Their fused K1* (l2 norm + gate + Gram + solve + w/u in one kernel) is
   documented in `kernels.py` as ~1.5x slower than the two-stage split on
   910B3, and the measurement here agrees (11.09 ms vs 7.53 ms).
5. *No output permute.* Triton stores `o` in its public layout directly, while
   we pay aclnn layout kernels per call: `pack_tokens` x4 = 0.259 ms, the
   output `permute().contiguous()` = 0.113 ms, the beta permute 0.015 ms,
   0.39 ms total (1.7% of the pass).  This is the one place where the
   reference is structurally cheaper: `kda_k2_outstate_kernel` could store its
   16x64 rows straight into `[B, T, H, D]` (`DataCopyParams(16, 4, 0, 508)`),
   and the four `pack_tokens` copies disappear if the K1 kernels address the
   public layout with a row stride of `H * D`.

## Related measurement

`torch.npu.NPUGraph` capture of the whole `separated` pass is numerically
exact (output/state delta 0.0) but gives **0.98x** (22.56 ms regular vs
23.01 ms replay), so the pass is not host-launch-bound and the 2048 launches
are not the thing to attack.
