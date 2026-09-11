# AscendC v1 kernel inventory (`kernels/v1/`)

`python/kda_ascendc_v1/api.py` compiles these sources at runtime (RTC) and
launches them through `launch_argsarray_engine`. Every producer/consumer pair
below has a fixed buffer layout; changing one side of a pair without the other
silently produces wrong numbers, so the contracts are spelled out here.

## K1: preprocess -> gram -> W/U solve

| Source | Kernel | Output |
|---|---|---|
| `preprocess.cpp` | `kda_preprocess_kernel` | `Qn`/`Kn` (bf16, packed `[c,16,128]`), `Gate`, `Gc`, `Beta`, `Decay`, `Qg`, `Kg`, `Rk`, `Rv` |
| `k1_gram.cpp` | `kda_gram_kernel` | `Aqk32`/`Aqk` bf16 `[c,16,16]`, `L` |
| `k1_solve_wu.cpp` | `kda_solve_wu_kernel` | `A32`/`A16` |
| `k1_solve_wu_cube.cpp` | `kda_solve_wu_cube_kernel` | `W`, `U` |

The preprocess kernel indexes the packed chunk-major layout, so `api.py` passes
the `*_pack` tensors (`pack_tokens` / the beta permute), never the public
`[B,T,H,D]` views.

Row reductions inside these kernels use the tested idiom
`Add(tmp, tile, tile[64], 64, M, BinaryRepeatParams(1,1,1,16,16,16))` followed by
`WholeReduceSum(rs, tmp, 64, M, 1, 1, 16)`: the fp32 L1 mask is 64 lanes, and
with `dstRepStride = 1` the results land contiguously at `rs[i]` (not `rs[i*8]`).

## K2: d12 -> vnew -> d3 -> d4 -> out/state

| Source | Kernel | Blocks | Output layout |
|---|---|---|---|
| `k2_init.cpp` | `kda_k2_init_kernel` | `BH*NV` | `s32`/`s16` `[task,64,128]` |
| `k2_d12.cpp` | `kda_k2_d12_kernel` | `BH*NV` | `d1`,`d2` `[task,chunk,16,64]` fp32 |
| `k2_d12_cube.cpp` | `kda_k2_d12_cube_kernel` | `BH*NV` | same as `k2_d12` |
| `k2_vnew.cpp` | `kda_k2_vnew_kernel` | `BH*NV` | `vnew` `[task,chunk,16,64]` bf16 and `vnew_t` `[task,chunk,64,16]` bf16 |
| `k2_kg_transpose.cpp` | `kda_kg_transpose` | `C` | `kg_t` `[c,128,16]` bf16 (K-major operand) |
| `k2_d3_cube_bv64.cpp` | `kda_k2_d3_cube_bv64` | `BH*NV` | `d3` `[task,chunk,16,64]` fp32 |
| `k2_d34.cpp` | `kda_k2_d34_kernel` | `BH*NV` | `d3` plus the task's 64 `d4` rows |
| `k2_d4_only.cpp` | `kda_k2_d4_only_kernel` | `BH*NV` | the task's 64 `d4` rows |
| `k2_d4_full.cpp` | `kda_k2_d4_full` | `BH` | the whole `[128,128]` `d4` tile |
| `k2_outstate.cpp` | `kda_k2_outstate_kernel` | `BH*NV` | `out_task`, `s32`, `s16` |
| `k2_outstate_full.cpp` | `kda_k2_outstate_full_kernel` | `BH*NV` | `out_task`, `s32`, `s16` |
| `k2_mix_d4_outstate.cpp`, `k2_mix_d12_vnew.cpp`, `k2_mix_all_cube.cpp` | MIX kernels | `BH` | same buffers as above |

### d4 layout contract

`d4[v][k] = sum_i v_new[i][v] * kg[i][k]` is always stored row-major as
`[v][k]` with a 128-float row stride, in one of two addressing modes:

- reused per head (`d4_reuse == 1`): region `[bh,128,128]`, written by
  `k2_d4_full.cpp` / the MIX kernels and read as `bh*D*D + iv*64*D`.
- per chunk (`d4_reuse == 0`): region `[c,128,128]`, written by
  `k2_d34.cpp` / `k2_d4_only.cpp` only for the rows owned by the block
  (`c*D*D + iv*64*D`) and read by `k2_outstate.cpp` at the same offset.

A block must never write the full `[128,128]` tile in the per-chunk layout:
with two v-tile blocks per head that both doubles the footprint and races with
the sibling block.

### Cube operand convention

`DataCopy(dst, src, Nd2NzParams(1, R, C, 0, C, R, 1, 0))` followed by a
non-transposing `LoadData2dParams` loads a row-major `[R,C]` tile as the L0
operand whose **rows are the n dimension and columns the k dimension**. That is
why d3 consumes `vnew_t` (`[v][j]`) and d4 consumes `kg_t` (`[k][i]`); the
non-transposed buffers cannot be fed to `Mmad` without an in-kernel transpose.

When the tile is exactly `C == 16` columns wide - one C0 block per row - the ND
layout *is* the NZ layout, so `Nd2Nz` degenerates to a plain copy and can be
replaced by a plain burst `DataCopy(dst, src, R * C)`. Do that: the conversion
still issues one 32-byte descriptor per row, which cost 7.3 ms per 512 chunks
in `k2_d34` while the whole kernel only needs 3.4 ms. The 128-wide staging
tiles (`W`, `Qg`, `S16`) do need `Nd2Nz` - a plain copy there silently loads
the wrong operand layout (verified: d12 output moves by 7e-2).

The B-operand rule above means a row-major `[16,128]` activation tile cannot be
fed to `Mmad` directly - the fractal rows must be the n dimension. `Nd2Nz`
cannot fix this on its own (it only moves whole 32-byte blocks, and swapping
`dstNzC0Stride`/`dstNzNStride` gave a 3.7 error), but the B1->B2 transposing
load can: `DataCopy(..., Nd2NzParams(1, 16, 128, 0, 128, 16, 1, 0))` stores the
tile as eight 16x16 fractals, and `LoadDataWithTranspose(b, lb,
LoadData2dTransposeParams(0, 8, 1, 0, 0))` transposes each of them into the zN
operand. `kda_solve_wu_cube_kernel` uses exactly this, so `rk`/`rv` keep their
natural layout and need no staging transpose.

### Performance notes

`[1,8192,32]` (16384 chunks, 512 chunk steps x 4 launches) is dominated by
per-launch and per-stage latency, not by FLOPs:

- The K2 chain is inherently sequential (`d12 -> vnew -> d34 -> outstate` all
  read or write the running state), so batching several chunks into one launch
  is *not* an option. Do not "optimize" the launch loop into a grid-stride
  loop over chunks: a multi-chunk version was measured 12% faster and produced
  stale-state results (2.4e-2 error).
- The chunk is 16 tokens (`CHUNK = 16`), so a `[1,8192,32]` pass runs 512
  chunk steps x 4 stage launches = 2048 launches. An empty kernel costs
  2.66 us of host time and 2.72 us of device time per launch at 64 blocks
  (the K2 stage width), so dispatch alone is ~5.5 ms of the ~14.1 ms K2 pass,
  while a whole `d12` launch is only 5.05 us of device time - the dispatch is
  as expensive as the work. Cutting the launch count is therefore the single
  biggest K2 lever left; do not spend it on fusing stages whose engines differ
  (the `mix_*` kernels lose 2-5x that way).
- Cost also tracks the *block* count: the same empty kernel takes 22.9 us per
  launch with 2048 blocks (~11 ns per block). Kernels that spawn one block per
  chunk (K1 `gram`/`solve`/`preprocess` and `kg_transpose`, 16384 blocks) can
  therefore amortise their per-block setup by handling several chunks per
  block - those stages have no cross-chunk dependency, unlike K2.
- Swapping the two 128-wide `Nd2Nz` loads in `k2_d12` for plain copies saves
  only 0.12 ms of its 3.02 ms (measured over the full 512-launch stage), so the
  earlier "Nd2Nz is 64% of d12" note does not reproduce; do not chase it.
- Batching several chunks per block in the K1 stages was measured and is *not*
  worth much: `k1_gram.cpp` rewritten to take `nchunk` (bit-exact against the
  one-chunk kernel) runs 2.36 ms at 16384 blocks, 2.27 ms at 8192, 2.21 ms at
  2048 and 2.24 ms at 512 blocks - a 7% ceiling. Removing the kernel's input
  loads entirely only reaches 2.23 ms and removing its stores 2.30 ms, so the
  stage is ~95% vector compute: the per-chunk cost is the 16-iteration row loop
  (5 vector ops and 4-5 `PipeBarrier`s per row, plus one `WholeReduceSum`),
  which is latency-bound rather than load-bound. Do not expect much from
  moving the Gram matmuls to the Cube either: the whole 16-row loop is 1.30 ms
  of the 2.35 ms (the `K` half alone is 0.93 ms, the `A` half 0.38 ms) while
  the exponent/mask prep is already 1.04 ms, so a Cube split would spend
  roughly 0.9 ms of prep plus ~0.7 ms of Cube work plus a mask pass to save
  1.3 ms. The prep is the part worth attacking (it computes `2^gc` and
  `2^-gc` from scratch even though `preprocess` already materialises the gated
  `Qg`/`Kg`).
- Every returned output used to be scrambled in ``t`` and ``h``:
  ``out_task.view(bh, NV, nt, CHUNK, BV).permute(0, 2, 3, 1, 4)`` keeps
  ``bh = b * H + h`` as one merged dimension through the permute, so the tensor
  that is finally shaped ``[b, t, h, d]`` actually holds ``[b, h, t, d]``.  Ten
  of the twelve modes return through that line, so they were all wrong the same
  way: the bit-exact mode-vs-mode checks cannot see it and the reference
  comparison accepted it because a swap error has the same magnitude as the
  signal (4.7e-3 against a 4.6e-3 reference absmax).  Splitting ``b`` and ``h``
  before the permute moves every mode to 3.05e-5 against the Triton reference
  while the cross-mode agreement stays bit-exact; `tests/test_output_layout.py`
  now fails on either mistake.
- Two layout optimizations that the Triton comparison suggested were measured
  and reverted.  Storing ``o`` straight into ``[B, T, H, D]`` from the outstate
  kernel turns each 16x64 tile into 16 rows of 128 B at an 8 KB stride and cost
  3.8 ms over the 512 launches (k2 stage 14.0 -> 17.8 ms) against the 0.11 ms
  the host permute costs.  Reading the public layout in ``preprocess`` removes
  the four ``pack_tokens`` copies (0.26 ms) but adds ~0.15 ms of strided load
  to the stage and the remainder is inside the +/-0.5 ms run-to-run noise of
  this device, so it was reverted too.
- The Triton reference compiles to a 16384-block MIX K1 (11.09 ms) and a
  64-block, single-launch K2 that keeps the fp32 state resident for all 512
  chunks (18.79 ms); both are slower than the `separated` path here (7.53 ms /
  13.95 ms), see `docs/TRITON_CODEGEN_REVIEW_20260911.md`.  The one place the
  reference is cheaper is layout: it stores `o` in the public layout, while we
  pay 0.39 ms of aclnn permute/copy kernels per pass.
- `persistent_scan_cube` is *not* a persistent kernel: `api.py` remaps the name
  onto the fused per-chunk `kda_k2_mix_all_cube`, which still launches once per
  chunk (512 launches instead of 2048). A single-kernel MIX loop
  (`k2_persistent_scan_cube.cpp`) was deleted: it never returned on this CANN
  runtime (killed after 150 s at `[1,8192,32]` and after 100 s at `[1,64,2]`),
  so the earlier "3504 ms -> 79 ms for the mode" note compared the dead kernel
  with the live `mix_all_cube` path.  See
  `docs/VLLM_ASCEND_KDA_REVIEW_20260911.md` for the cross-core flag rules that
  loop violated and for the schedule to use when it is restarted.
- `mix_all_cube` was 3.5x *slower* than `separated` (79.2 vs 22.7 ms at
  `[1,8192,32]`) only because it never got the idioms the separated kernels
  had.  Porting them (one `Fixpipe` per tile, burst loads for 16-column
  operands, a fused d34 AIC stage, UB `Transpose` for `v_new^T`, two-repeat
  `Mul` for the state update) takes the mode to 24.75 ms against `separated`
  22.67 ms with identical numerics.  Fusion still loses, but by 9% rather than
  3.5x, and the lesson is that the 2048-launch chain is not launch-bound: per
  chunk the separated path spends 27 us and the fused path 32 us for a few
  microseconds of matmul.  `mix_aic_1_2` (36.3 ms) and `mix_d12_vnew` (74.6 ms)
  still carry per-fractal `Fixpipe` calls and have not been ported.
- Each 16-row `Fixpipe` is ~6 us per launch; batch per tile, but check the
  result - a single 64x128 `Fixpipe` with `srcStride = 16` silently produced a
  wrong tile (3e-1 error).
- The K1 `solve` kernel used to be the largest single kernel (~6.5 ms): a
  scalar forward substitution plus a vector matvec whose per-row scalar
  broadcasts dominated. `w = A_inv @ rk` / `u = A_inv @ rv` now run on the
  Cube (`kda_solve_wu_cube_kernel`), which took the stage from 5.9 ms to
  3.3 ms; `W` is bit-identical to the vector path and `U` moves by 2.4e-4.

## Verification

```bash
python tools/compile_all_server.py                     # RTC compile of every kernel
python -m pytest tests/test_persistent_scan.py tests/test_persistent_scan_cube.py
python tests/test_mix_aic_1_2.py                       # script-style checks
python tests/test_d12_cube.py
python tests/test_s15_mix_d12_vnew.py
python tests/test_cube_separated.py
```

`tests/test_cube_separated.py` compares `separated` against `cube_separated`
for `[1,32,2]`, `[2,1024,4]`, `[2,4096,8]` and `[1,8192,32]`; both must be
finite and agree within 1e-3 / 1e-4. All ten `k2_mode` values are also checked
against the Triton reference (`src/kda_bt16`) on `[1,64,2,128]` with
`state_v_first=True`.

## Reproducibility note

The persistent kernels keep the cross-chunk state in UB and update it with
vector ops, then read it back with the scalar unit on the next iteration. That
vector -> scalar dependency needs an explicit `SetFlag/WaitFlag<HardEvent::V_S>`
after the update; without it the final state drifts by ~1 ulp between runs and
`tests/test_persistent_scan.py` (which requires bit-exact agreement between
`persistent` and `persistent_scan`) fails intermittently.
