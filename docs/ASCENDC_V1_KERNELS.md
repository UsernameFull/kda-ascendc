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
| `k1_pre_gram.cpp` | `kda_pre_gram_kernel` | everything the two rows above write, in one block per chunk; reads `q`/`k`/`v`/`g` in the public `[B,T,H,D]` layout |
| `k1_solve_wu.cpp` | `kda_solve_wu_kernel` | `A32`/`A16` |
| `k1_solve_wu_wide.cpp` | `kda_solve_wu_wide` | the same, 32 chunks per vector instruction |
| `k1_solve_wu_cube.cpp` | `kda_solve_wu_cube_kernel` | `W`, `U` |

The preprocess kernel indexes the packed chunk-major layout, so `api.py` passes
the `*_pack` tensors (`pack_tokens` / the beta permute), never the public
`[B,T,H,D]` views.  `kda_pre_gram_kernel` is the exception: it reads
`q`/`k`/`v`/`g` out of the public layout with a byte-strided `DataCopyPad`
(`xRowBytes = H*D*2 - D*2`, `gRowBytes = H*D*4 - D*4`), which is what lets
`api.py` skip `pack_tokens` for the four stage-1 inputs; everything it writes
is still packed.

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
| `k2_persistent_loop.cpp` | `kda_k2_persistent_loop` | `ceil(BH/2)` | one launch for all `NT` chunks (see below) |

### `persistent_loop`: the whole K2 recurrence in one launch

`k2_mode="persistent_loop"` runs `d12 -> vnew -> d34 -> outstate` for every
chunk inside a single `KERNEL_TYPE_MIX_AIC_1_2` launch, so the host-side chunk
loop in `api.py` collapses to one `kda_kg_transpose` plus one
`kda_k2_persistent_loop` call. Each block owns up to `MAXH = 2` heads
(`h = blk, blk + nblk, ...`) and each AIV subcore owns one 64-column value
tile, so a block has `nh * 2` state tiles in flight and the AIC is never
blocked behind its own chain:

```
AIC: [wait R; d12(h); set C1] x nh   then   [wait V; d34(h); set C2] x nh
AIV: [wait C1; vnew(h); set V] x nh  then   [wait C2; out(h); set R] x nh
```

Both AIV subcores execute the identical flag sequence (the head is the only
unit both subcores share; only `iv = GetSubBlockIdx()` differs). Four flag ids
`0..3` carry the loop, the protocol is depth one and iteration-invariant, and
the AIV pre-sets `R` `nh` times before the loop so the first chunk needs no
special case. The fp32 state lives in UB for the whole loop (loaded from `H0`,
stored to `S32` once at the end); only its bf16 copy `S16` goes to GM each
chunk for the Cube operands. `api.py` must therefore allocate
`d4_full` as `[bh,128,128]` (per-head reuse) and pass `nblk` with
`nblk * MAXH >= bh`, otherwise the head map is not total.

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
- K1 was retuned stage by stage and the pass went 14.00 -> 9.35 ms at
  `[1,8192,32]` (`separated` 22.62 -> 18.16 ms) with the error against Triton
  *unchanged to the digit* at every shape (`[1,8192,32]`: out 6.10e-05,
  state 4.12e-04; `[2,4096,8]`/`[3,2048,8]`/`[1,1024,32]`: out 6.10e-05).
  In-pipeline profile (`KDA_PROFILE=1`), HEAD -> now:
  preprocess+gram 1.94 + 2.36 -> `pre_gram` 2.24, solve 3.29 -> 0.65 (AIV
  2.44 -> 0.97 -> 0.10 + Cube 0.90 -> 0.55), K2 incl. `kg_transpose` 5.42 ->
  4.71; `total_ms` 13.01 -> 7.43.  Isolated stage times on the current tree
  (`/tmp/kdaval/k1_one.py all`, best of 5): preprocess 1.96, gram 1.74,
  solve AIV 1.01 -> 0.10 (wide kernel, `check_solve.py`), `kg_transpose`
  0.16, solve Cube (NC=4) 0.55 ms; the same
  harness on the *fused* kernel gives 2.34 ms.
  - `k1_gram.cpp` only builds the lower triangle (`rows = i + 1` passed to
    `Mul`/`MulAddDst`/`WholeReduceSum`, 136 row-passes instead of 256), uses
    `MulAddDst` for the second half product (the separate `Add` is gone) and
    clears `redA`/`redK` with one `Duplicate` each, because an entry above the
    diagonal is never written and the mask multiply would propagate a stale
    NaN or not, and the whole stage went 2.36 -> 1.71 ms; the three changes
    were applied together, so no per-change split is claimed here.
    Measured and rejected: dropping the in-loop `PipeBarrier<PIPE_V>`s (no
    change - the vector pipe is issue-bound, not latency-bound) and replacing
    the second `Exp` with `Reciprocal` (saves 0.05 ms and drifts the output).
  - `k1_solve_wu.cpp` no longer inverts `L` with a scalar recursion.  It now
    runs the row recursion `A_inv[i] = e_i - sum_{j<i} L[i][j] * A_inv[j]` on
    the vector unit (`Axpy(af + i*M, af + j*M, -cij, M)`, one barrier per
    step), 8 chunks per block: 2.44 -> 0.97 ms, bit-identical to the old
    scalar solve over 64 chunks.  `A_inv` has to start from the *identity*
    (starting from `I - L` double counts the linear term).  A `Div` variant
    and a batched-but-still-scalar variant were both wrong or no faster.
  - `k1_solve_wu_wide.cpp` is the same recursion with the *chunk* axis as the
    vector lane width instead of the 16-element row: one `MulAddDst` updates
    row `i` of 32 chunks at once, so the 120 steps of the recursion cost 120
    instructions per 32 chunks instead of 120 per chunk.  The three layout
    facts it rests on were checked on hardware with a probe kernel
    (`/tmp/kdaval/probe2.cpp`): `Brcb(dst, src, 2*NC, BrcbRepeatParams(1, 8))`
    reads one 32B block - eight columns of one chunk - per repeat and writes
    eight blocks, so block `16*ch + j` of the expansion holds `-L[ch][i][j]`
    eight times; a binary vector op with `src1BlkStride = 0` then reads that
    single block for both 32B halves of a 16-lane repeat, i.e. broadcasts one
    coefficient over exactly the sixteen lanes of one chunk's row, and
    `src1RepStride = 16` walks the chunks; and
    `DataCopyParams(NC, 2, 30, 0)` gathers two-block rows 32 blocks apart,
    turning the chunk-major `L` of GM into the per-row layout `Brcb` wants
    without any extra GM traffic (2 KB per row of 32 chunks).
    Measured against the per-chunk kernel, bit-identical on both outputs
    (`0.000e+00`, every bf16 value equal): NC=8 0.2235 ms, NC=16 0.1285 ms,
    NC=32 **0.1006 ms** vs 0.9120 ms - 9.1x.  NC=64 does not fit (the four
    live tiles would need ~190 KB of the 192 KB); what is left at NC=32 is
    memory traffic, 40 MB per pass (16 MB `L` read, 16 MB fp32 + 8 MB bf16
    `A_inv` write) against ~370 GB/s, so it sits at the bandwidth floor.
    In-pipeline this is `solve_ms` 1.47 -> 0.65 ms and `total_ms`
    8.41 -> 7.60 ms at `[1,8192,32]`.
    A two-accumulator split of the recursion (to break the 120-deep dependency
    chain) was *slower* (0.2497 vs 0.2235 ms at NC=8): the chain is not the
    limit, the per-instruction issue cost is.  That is also why NC=16/32 win
    even though the grid gets smaller.
    `api.py` allocates `L`/`A32`/`A16` rounded up to `SOLVE_WIDE_NCHUNK`
    chunks so that the last group's gather and stores stay inside an
    allocation; the padded chunks are never read (the debug dict hands out
    `[:c]` views) and, unlike a `zero_()` of the tail, the padding costs no
    extra host launch - measured at `[1,32,2]`, where padding a 4-chunk shape
    out to a 32-chunk block is 7/8 wasted work, the call is still 0.334 ms
    (`persistent_loop`) against 0.342 ms before.
  - `k1_solve_wu_cube.cpp` keeps 4 chunks in flight (WU_NCHUNK=4, one AIC
    block per 4 chunks): 0.90 -> 0.55 ms, and NC=1 is 0.88 ms, i.e. the win is
    the per-block fixed cost (~1.26 us), not the mmad.  NC>=6 is faster
    (0.33/0.26 ms) but wrong on this hardware: the L0A/L0B slots are indexed
    `pass*NC + ch` and the L0C/`qc` queue is only `NC` deep, so the mmad of
    chunk *k* overlaps the L0A/L0B write of chunk *k+1* and L0C is rewritten
    under a live `Fixpipe`.  Keep NC <= 4.  Each pass must also re-issue its
    own B load (`Rk` -> `W`, `Rv` -> `U`); an earlier version that loaded both
    only once dequeued an empty queue for the second pass and produced a `U`
    that was 100% wrong while `W` stayed correct - `val_k1.py` catches it in
    one run, `tools/bench_modes_vs_triton.py` did not, because every mode
    consumes `U` through the same wrong tensor.
  - `k2_kg_transpose.cpp` (K1-adjacent, it is launched inside the K2 stage)
    now moves `NCHUNK = 8` chunks per block with one strided
    `DataCopyParams(M, 1, 7, 0)` gather per 16x16 tile instead of 16 row
    gathers: 0.72 -> 0.16 ms, bit-identical to `torch.transpose`.
  - `k1_pre_gram.cpp` reads its four stage-1 inputs (`q`/`k`/`v`/`g`) straight
    out of the public `[B, T, H, D]` layout, so `api.py` no longer calls
    `pack_tokens` on them at all.  One token is `D` contiguous elements and
    consecutive tokens of a chunk sit `H*D` elements apart, which is exactly the
    gather `DataCopyPad`'s byte-stride form expresses; a `DataCopy` block/stride
    load of the same gather silently drops most of the bursts on this part (see
    the note above).  End to end at `[1,8192,32]`: 7.482 -> 6.926 ms (median of
    5, same process settings: 7.461 min before, 6.861 after), i.e. the pack's
    1.07 GB round trip is gone and only the read side of it remains; at
    `[2,1024,4]` 0.814 -> 0.762 ms.  `out`, `state` and all thirteen
    intermediates stay bit-identical (`0.000e+00` on every one, two shapes).
    The `[c, M, D]` addressing of everything downstream is unchanged, so
    `pbQn`/`Kn`/`Gate`/`Gc`/`Decay`/`Aqk`/`L` and the K2 layout still hold.
  - `k1_pre_gram.cpp` fuses stages 1+2 into one AIV block per chunk (`api.py`
    now launches it instead of `kda_preprocess_kernel` + `kda_gram_kernel`
    back to back): the Gram half reads `Qn`/`Kn`/`Gc` straight out of UB and
    the 32 KB/chunk GM round trip is gone.  In-pipeline at `[1,8192,32]`:
    1.91 + 1.71 = 3.62 -> 3.08 ms; isolated (`/tmp/kdaval/pregram_one.py`,
    best of 5) 3.675 -> 3.114 ms at `[1,8192,32]` and 1.866 -> 1.591 ms at
    `[1,4096,32]` (1.18x / 1.17x), bit-identical on all 13 outputs (every
    ``max(abs(diff))`` exactly `0.000e+00`).  The GM copies of
    `Qn`/`Kn`/`Gate`/`Gc` are now skipped unless the caller asked for the
    intermediates, so the fast path never writes them at all.
    This is the right lever because `preprocess` is DMA/fixed-cost bound, not
    FLOP bound: measured decomposition of the *pre-fusion* stage - an empty
    kernel costs 0.22 ms and a no-`Exp` stub 1.91 ms of the 1.96 ms, so the
    two `Exp` passes are free and the 15-step cumsum is 0.07 ms; ~140 vector
    instructions per block (~80 of them the per-row `Muls`/`Sub`/`Duplicate`
    passes) against a ~1 us/block fixed cost is where the rest is.
    - Per-row scalars are broadcast with `Brcb` instead of a 16-iteration
      `Muls`/`Duplicate` loop that read each scalar back with `GetValue`.
      `Brcb(dst, src, 2, BrcbRepeatParams(1, 8))` turns the 16 values into a
      tile where every row owns eight 32B blocks; the consumer then runs 64
      lanes x 16 repeats with `BinaryRepeatParams(1, 1, 0, 16, 16, 1)` (repeat
      strides are in 32B blocks, so `src1RepStride = 1` walks one scalar per
      repeat while `dstRepStride = 16` walks the 128-lane rows), once for the
      low half and once for the high half of each row.  Applied to the l2 norms
      (q and k), `beta` (also consumed by the Gram's `K` half, which is now
      pre-scaled once instead of per row) and the chunk-centred gate, this
      removes 68 vector instructions and 64 scalar `GetValue` reads per block:
      3.114 -> 2.679 ms at `[1,8192,32]` and 1.590 -> 1.373 ms at
      `[1,4096,32]`, bit-identical on all 13 outputs (`0.000e+00`).
      Two things that do *not* work here: a constant operand in the first
      position with `src0RepStride = 0` (silently wrong, unlike the
      `src1RepStride = 0` broadcast the bias add already uses), and
      `BrcbRepeatParams(8, 64)` for a 64-copies-per-row tile.  The kernel was
      re-profiled after the change (`msprof --aic-metrics=PipeUtilization`,
      `/tmp/kdaval/prof_pregram.py`): 3079 -> 2707 us, `aiv_vec_ratio`
      0.716 -> 0.686, `aiv_scalar_ratio` 0.309 -> 0.223, `aiv_mte2_ratio`
      0.104 -> 0.119, total AIV cycles 265.6M -> 233.5M, so the win is real
      issue-bound work and the stage is still VEC bound (69%).
    - The fused body is issue-bound, not FLOP-bound, and three latency
      changes follow from that (`msprof --aic-metrics=ArithmeticUtilization`
      on the fused kernel: `aiv_vec_fp32_ratio` 0.20 while `aiv_vec_ratio` is
      0.686, i.e. the vector pipe is *engaged* 69% of the block but doing fp32
      math only 20% of it; a micro-benchmark of bare instructions puts a fixed
      ~25-35 cycles on every vector instruction plus ~1.1 cycles per repeat
      for elementwise ops and ~7 for each repeat of a `WholeReduceSum`):
      2.679 -> 2.593 ms at `[1,8192,32]`, still bit-identical on all 13
      outputs.
      - Every input load is now issued up front behind its own
        `MTE2 -> V` event and the matching `WaitFlag` sits at the consumer:
        `Q` gates the q norm, `K` the k norm, `G`+`Beta`/`Alog`(+the two
        triangular masks, also hoisted) the gate section, `V` the `rv` chain.
        The block used to issue five copies and then wait for *all* of them,
        so the two l2 norms were stalled behind `G` (8 KB) and `V` (4 KB).
      - `Qg`/`Kg`/`Rk`/`Rv`/`BetaOut` are stored as soon as they exist
        instead of in the final store block, so ~16 KB/block of MTE3 drains
        behind the Gram loop.
      - The two `V_S` sync pairs around the l2 norms were dead (the norms were
        read back with `GetValue` before the `Brcb` rewrite) and cost ~2%; the
        pair that *is* live is the one around `Exp(alog)`.
      Measured and rejected in the same round: dropping all 48
      `PipeBarrier<PIPE_V>()` (2.72 vs 2.69, i.e. the barriers are ~free),
      `Muls(1.0)` for every `Exp` (2.66, so the five `Exp` passes are worth
      only 0.03 ms), and `Muls` for `Rsqrt` (no change).  Removing every
      input `DataCopy` and every store of the pack chains are the two big
      levers that remain: 2.69 -> 2.49 and 2.69 -> 2.39 ms respectively.
    - One block walks `unroll` consecutive chunks (the body is byte-identical,
      just wrapped in a runtime-bound loop; `api.py` picks `unroll =
      clamp(c // 512, 2, 8)`, or 1 below 256 chunks).  Every block pays a fixed
      setup cost worth ~10% of a chunk's work, and back-to-back launches (host
      overhead hidden) put the optimum at >= 256 blocks: 2.593 -> 2.342 ms at
      `[1,8192,32]` (unroll 8, 2048 blocks), 0.333 -> 0.308 ms at
      `[1,1024,32]` (unroll 4, 512 blocks), 0.0953 -> 0.0932 ms at
      `[2,1024,4]` (unroll 2, 256 blocks), all bit-identical, while unroll 8
      at `[2,1024,4]` (64 blocks) *loses* 16% to wave quantization.  Under 256
      chunks the loop wrapper itself costs ~6% and the policy keeps unroll 1.
      In-pipeline this is `pre_gram` 2.59 -> 2.31 ms, `total_ms` 8.83 -> 8.48.
    - A last pass deleted four more instructions per chunk, all bit-identical
      (`0.000e+00` on all 13 outputs): the `Muls(gf, gf, aexp)` +
      `Muls(gf, gf, -1)` pair folds into one `Muls` with the scalar negated
      (sign flips are exact), `Sub; Muls(-1)` for `gate_last - gate` becomes
      one `Sub` with its operands and repeat strides swapped, the A-side
      triangular mask multiply is a no-op because the Gram loop only writes
      the lower triangle and `Duplicate` already zeroed the rest, and the
      dead `MaskS` tile fetch goes with it (`MaskL` has to stay: the K-side
      loop does write L's diagonal and the mask is what zeroes it).
      Back-to-back 2.342 -> 2.258 ms at `[1,8192,32]`; in-pipeline
      `pre_gram` 2.31 -> 2.29 ms, `total_ms` 8.48 -> 8.41.
      The same round pinned the cost model that motivates counting
      instructions here: +8 one-repeat `Adds` per chunk cost +0.040 ms
      (~27 cycles per vector instruction), +8 16-repeat `Mul`s +0.056 ms
      (~0.6 cycles per extra repeat), +8 one-repeat `Exp` +0.051 ms (the
      special-function unit is dearer per repeat but not dominant), and a
      512-instruction loop of 64-lane `Mul`s runs 4.8x slower than the
      same work in 8 instructions of 16 repeats.  Measured and rejected in
      the same round: dropping the 15 cumsum and/or the 64 Gram-loop barriers
      (no change, the vector pipe is in-order so they are free), giving the
      Gram loop a second product tile so its A and K halves stop serialising
      on one buffer (no change: issue-limited, not latency-limited), 256B UB
      padding on `gb`/`redA` (4.5%/4.2% *worse*)/`ga`/`gtb` (neutral), unroll
      4/16/32 (worse than 8), and moving the Gram to the Cube (24 KB per
      chunk of extra GM traffic each way for ~4.7k of ~13.7k core cycles, a
      wash at ~370 GB/s).
    - The one `PipeBarrier<PIPE_ALL>` left on the fused fast path guards the
      `Decay` store, because `t2` is reused by the `qg` product a few
      instructions later; that needs only the MTE3->V half of the barrier
      (`SetFlag<HardEvent::MTE3_V>` after the copy, `WaitFlag<HardEvent::MTE3_V>`
      before the `Mul` that overwrites the tile).  Back-to-back 2.259 ->
      2.209 ms at `[1,8192,32]` (R=10, unroll 8, bit-identical on all 13
      outputs); in-pipeline `pre_gram` 2.285 -> 2.237 ms, `total_ms` 7.463 ->
      7.433.  Measured and *not* shippable: deleting the tail
      `PipeBarrier<PIPE_ALL>` as well is worth another 0.125 ms (2.259 ->
      2.134 ms) but leaves the next chunk's `Q`/`K`/`V` loads racing the
      previous chunk's `Qn`/`Kn`/`Rv`/`Aqk32`/`L`/`Aqk16` stores, and every
      safe formulation of that sync hung the device on the *first* launch:
      loop-carried `MTE3->MTE2` flags, and loop-carried `MTE3->V` flags waited
      either at the top of the body or at the first write to a stored tile,
      with or without the MTE3->MTE2 pair.  The same pattern runs in a toy
      micro-kernel, the compiled code is the same size, and in-body flag pairs
      (the `Decay` one above) work, so this is a property of the loop-carried
      flag on this runtime rather than of the direction; the way to collect
      that 0.125 ms is to make the loads and the stores stop sharing UB (a
      double-buffered `TQue`), not to look for another flag.
    - The fused body is *compiler fragile* and must not be reformatted: the
      same arithmetic written slightly differently trips `aivec error` (mte
      error info `0x8030860ef`, "address for the scalar to access the internal
      buffer of AICore is out of bounds") at runtime, with no compile-time
      diagnostic from `rtc_compile`.  Confirmed triggers: moving a
      `PipeBarrier<PIPE_V>`, renaming one of the buffers, and re-wrapping the
      `TBuf`/`InitBuffer` declaration lists.  Extra unused arguments and
      unused buffers are fine; the ~236 KB of live UB is not the problem
      (the same kernel allocates with 10 dummy int args added).  Keep the
      file's layout.
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
  the host permute costs.  The other one - reading the public layout instead of
  the packed ``[c, M, D]`` copies - was retried later and *did* pay off, but
  only through ``DataCopyPad``: see the ``pack_tokens`` note under K1.
- ``DataCopy``'s block/stride form cannot read a strided GM source on this part
  (910_9382 / CANN 9.1.0).  A probe kernel that gathers 16 rows of 256 B with a
  non-zero ``srcStride`` (``DataCopyParams(16, 8, 40, 0)``, ``gap=8`` in
  ``/tmp/kdaval/probe_stride4.py``) returns rows 0-3 correct, row 4 stale, rows
  5-7 correct and rows 8-15 untouched - i.e. some half of the bursts never
  lands, and the larger the gap the fewer rows arrive (``gap=40`` writes rows
  0-2 of 16).  ``DataCopyPad`` with the byte-unit ``DataCopyExtParams(16, 256,
  gap_bytes, 0, 0)`` reads the identical gather correctly for every gap tested,
  so the strided input path uses that form.  Worth checking before any future
  strided-row load.
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
- The device-side loop is back as `persistent_loop` and it wins: `[1,8192,32]`
  22.55 -> 13.67 ms for the whole pass and 14.03 -> 5.29 ms for K2 alone
  (2.65x); `[2,4096,8,128]` 6.92 -> 4.50 ms (K2 6.85 -> 2.29 ms, 2.99x).
  Output and
  state are bit-identical to `separated` (`0.0` / `0.0`) at every shape tested
  and K2 is now one launch instead of 2048. What remains of the pass is K1
  (`pre_gram` 2.29 + solve 0.65 + `kg_transpose` 0.16 ms, after the K1 retune
  below), so the next lever is no longer the K2 launch count.
  - The loop only became reliable once the AIC keeps a `PipeBarrier<PIPE_ALL>`
    after each stage's `FIX_M` wait, exactly like the per-chunk
    `run_d12_aic`: without them the kernel faults with an aicore exception
    (`FIXP`/`MTE`/`CUBE` error registers, PC in the `mix_aic` half) at
    `[1,8192,32]` in roughly one launch in three, while `[1,64,2]` and
    `[1,1024,32]` pass. The barriers cost nothing measurable (13.67 vs
    13.52 ms without them), so they stay.
  - The flag schedule itself was verified in isolation with a data-free probe
    kernel (`512` iterations x `16` blocks x `nh = 2`, five launches, 2 ms
    each) before blaming the protocol: four ids, two heads per block, both AIV
    subcores setting and waiting the same ids is stable on this runtime. The
    stall of the deleted loop was not a property of "long cross-core loops" as
    such.
  - `nblk` is chosen per shape: one head per block while `bh <= 24` (the AIC
    count) is ~4% faster (`[2,4096,8]` 4.55 -> 4.38 ms), and two heads per
    block above that, because 32 blocks on 24 AICs queue a second wave
    (`[1,8192,32]` 13.8 -> 17.2 ms).  `KDA_PERSIST_LOOP_BLOCKS` overrides it
    for experiments; the kernel requires `nblk * 2 >= bh` or the head map is
    not total.
  - The `d4` store is one `128 x 128` `Mmad` plus one `Fixpipe` per head and
    chunk instead of eight `16 x 128` pairs.  A `Fixpipe` whose `mSize`
    exceeds 16 walks L0C as one contiguous run of `16 x 16` C0 fractals, which
    is exactly the layout a *single* `Mmad` writes, so `srcStride = mSize` is
    the rule describing that walk; the eight `Mmad`s of the old form each
    wrote their own m-major band of the same region, and no `srcStride` can
    read that back (a probe kernel - one-, two-, four-band and single-`Mmad`
    dumps against the eight-call reference - is bit-exact only for the
    single-`Mmad` form).  Worth 8.41 -> 7.51 ms end to end at `[1,8192,32]`
    (median of 30 interleaved runs: 7.58/8.42/8.36 per round before,
    7.51/7.50/7.54 after) even though the isolated device time barely moves
    (`msprof` op summary: K2 3.59 -> 3.53 ms): the eight-call form falls into a
    second mode where `k2_ms` sits at 4.5-4.7 ms instead of 3.7 ms and the
    merge removes it.  Output and state stay bit-identical (`0.000e+00`).
  - The AIV's `v_new^T` build (64 one-block `DataCopy` gathers + four 16x16
    `Transpose`s per head and chunk) looks like the obvious next target and is
    *not* worth attacking: stubbing the gather, the transpose and the store out
    of the loop moves the pass from 13.67 to 13.66 ms. The 64 tiny MTE ops are
    fully hidden behind the AIC's `d34`, so any replacement would be a rewrite
    for nothing. Measured, reverted, recorded here.
  - The AIV's `v_new = u - d1` row loop is one instruction instead of sixteen
    (`Sub(vf, vf[iv*BV], d1, BV, M, BinaryRepeatParams(1, 1, 1, 8, 16, 8))`:
    the dst and the `d1` operand step by one 64-float row, the `u` operand by
    one 128-float row).  Unlike the gather above this one is on the critical
    path: K2 alone 4.666 -> 4.556 ms (three runs each) at `[1,8192,32]`
    (`KDA_PROFILE=1`), bit-identical output, so `total_ms` 7.60 -> 7.46.
    Merging the 64 staging `DataCopy`s of the same stage into four strided
    ones (measured
    together with it at 4.535 ms) is inside the +/-0.02 ms run-to-run noise of
    the stage, so it was reverted.
  - A loop needs the *vector* side's WAR drains too, and they are not free to
    skip: every iteration reuses the same UB staging buffers (`ub`/`vf`/`vb`/
    `sc` in stage 2, `d2`/`d3`/`d4`/`dec`/`ob`/`s16` in stage 4) while the
    previous iteration's reads and MTE3 copies out of them may still be in
    flight. Without a `PipeBarrier<PIPE_ALL>` after each `CrossCoreWaitFlag`,
    the *output* drifts by ~6e-3 at `[1,8192,32]` while the state stays
    bit-identical, and the first launch of a process differs from all later
    ones (the staging buffers are `torch.empty`). The one-chunk-per-launch
    kernels can never show this. Cost of the two drains: 13.66 vs 13.67 ms,
    i.e. nothing, because the `S16`/`Out` MTE3 copies already gate the flag
    that follows them.
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
  wrong tile (3e-1 error).  The reason is now known: a merged tile is walked as
  contiguous C0 fractals, i.e. `srcStride = mSize`, so the 16-row stride is
  only right when `mSize == 16`.  See the `d4` note under `persistent_loop`.
- The K1 `solve` kernel used to be the largest single kernel (~6.5 ms): a
  scalar forward substitution plus a vector matvec whose per-row scalar
  broadcasts dominated. `w = A_inv @ rk` / `u = A_inv @ rv` now run on the
  Cube (`kda_solve_wu_cube_kernel`), which took the stage from 5.9 ms to
  3.3 ms; `W` is bit-identical to the vector path and `U` moves by 2.4e-4.

## Verification

```bash
python tools/compile_all_server.py                     # RTC compile of every kernel
python -m pytest tests/test_persistent_scan.py tests/test_persistent_scan_cube.py
python -m pytest tests/test_persistent_loop.py           # single-launch device loop
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
