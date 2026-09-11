# How vllm-ascend implements KDA, and what we can borrow

Date: 2026-09-11.  Checkout reviewed: `/data/s00977099/vllm_hub/vllm-ascend`
(v0.22.1rc1, head `5f6faa0c`).  Our measurements are on NPU 1, shape
`[1, 8192, 32, 128]`.

## 1. What they run

`vllm_ascend/ops/triton/fla/chunk.py::chunk_gated_delta_rule_fwd` is the whole
forward, with `chunk_size = 64`:

1. **K1 stays in Triton**: `chunk_local_cumsum`, `chunk_scaled_dot_kkt_fwd`,
   `solve_tril`, `recompute_w_u_fwd`.  Then five explicit
   `.transpose(1, 2).contiguous()` conversions turn `k`, `w`, `u`, `g`, `q`
   from `[B, T, H, *]` into `[B, H, T, *]`, i.e. they pay the layout tax inside
   the model rather than fusing it.
2. **The state scan is a native op**: `torch.ops._C_ascend.
   chunk_gated_delta_rule_fwd_h(k, w, u, g, initial_state, chunk_size=64,
   save_new_value=True, use_exp2=False, transpose_state_layout=False)`,
   returning `h` (every chunk's state), `v_new` and `final_state`.
3. **The output is a second native op**: `torch.ops._C_ascend.chunk_fwd_o(q, k,
   v_new, h, scale, g, chunk_size=64)` = `o = q @ h + tril(q k^T) @ v_new`
   scaled.  Two more `.transpose(1, 2).contiguous()` conversions follow.

So `h` is materialised for the whole sequence because `fwd_o` is a separate
launch; our `outstate` stage fuses the same step and never writes `h`.

## 2. How their two AscendC kernels are built

Both live in `csrc/moe/{chunk_gated_delta_rule_fwd_h,chunk_fwd_o}/` and are
Catlass `tla` kernels with the same skeleton:

- `op_kernel/*.cpp` is a dtype dispatch onto
  `Catlass::Gemm::Kernel::GDNFwdHKernel<inputType, gType, stateType, fp32>`,
  declared as `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` (1 AIC : 2 AIV).
- `op_host/*_tiling.cpp` sets `SetBlockDim(aicCoreNum)` - *one launch for the
  whole sequence*.  The workspace holds per-core ping-pong buffers
  (`coreNum * chunkSize * vHeadDim * PING_PONG_STAGES` for `v_work`, the same
  for `v_new`, `coreNum * kHeadDim * vHeadDim * PING_PONG_STAGES` for `h`) plus
  16 MiB of reserve.
- The kernel (`op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp`, 407 lines) uses
  `L1TileShape = 128x128x128`, `MmadPingpongTlaMulti`, two `BlockMmad`
  (`w @ h` with both operands RowMajor, `k^T @ v` with `k` declared
  `layout::ColumnMajor`) and two AIV epilogues (`v_new`, `h += h_work`).
- The block scheduler (`block_scheduler_gdn_fwd_h.hpp`, 285 lines) walks the
  chunk loop **on the device** for one `(batch, v-head)` task per core, with
  `PING_PONG_STAGES = 2` and `headInnerLoop = 2`, i.e. two heads interleaved
  per core so that a core always has a second independent chunk chain to run
  while the first waits on its dependency.

The per-chunk critical path is a strict alternation:

```
AIC: wait vec2Done -> cube1: v_work = w @ h[i]  -> set cube1Done
     [iterId > 1] wait vec1Done -> cube2: h_work = k^T @ v_new -> set cube2Done
AIV: wait cube1Done -> v_new = u - exp(g_last - g) * v_work; store
                       v_new_decay = v_new * exp(g_last - g) -> set vec1Done
     [iterId > 1] wait cube2Done -> h[i+1] = h[i] * exp(g_last) + h_work
                                -> set vec2Done
```

Three details make that loop sound, and they are exactly what our deleted
`k2_persistent_scan_cube.cpp` did *not* do:

1. **The prologue is absorbed by pre-set flags**, not by a conditional first
   iteration: before the AIV loop they push
   `Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vec2Done)` twice, so the AIC's first
   two `CrossCoreWaitFlag(vec2Done)` calls retire immediately and the loop body
   stays iteration-invariant.  Our loop instead had `if (chunk > 0) { wait }`.
2. **One set and one wait per stage per iteration, strict alternation, depth 1.**
   No flag can ever be set twice before it is consumed.
3. **Flag ids 0..3.**  `catlass/arch/cross_core_sync.hpp` defines
   `FFTS_MAX_FLAG = 7` and reserves 8/9/10 for
   `AIV_INTER_BLOCK_BARRIER` / `AIC_INTER_BLOCK_BARRIER` /
   `AIV_INTER_SUBBLOCK_BARRIER`.  Our `k2_mix_*.cpp` kernels use 8/9/10 as
   their own sync ids (they work today because we never call the Catlass
   barriers, but it is a landmine to keep them), and the deleted persistent
   kernel used 8/10/13.

The same header also exposes `MAX_REVERSE_DEPTH = 16` and
`CrossCoreFlagWithReverse<16>`: in a long loop, every 16 sets must be repaid by
one wait on a reverse flag.  Any device-side loop that sets a flag every
iteration without that repayment can outrun the flag FIFO.  That is the
mechanism we most likely tripped over in the deleted loop (where, on top of the
conditional waits, the two AIV subblocks both set the `AIC`-directed flags each
iteration while the AIC consumed them once).

## 3. Same-shape comparison against our kernels

| | vllm-ascend | us (`separated`) |
|---|---|---|
| chunk size | 64 | 16 |
| K1 | Triton, 11.09 ms (see `TRITON_CODEGEN_REVIEW_20260911.md`) | AscendC, 7.53 ms |
| K2 | 2 launches per pass (`fwd_h`, `fwd_o`), device-side chunk loop | 2048 launches per pass (4 stages x 512 chunks) |
| K2 time | not measurable here (different chunk size/shape contract) | 13.95 ms |
| layout | 5 input + 3 output `transpose(1,2).contiguous()` | 0.39 ms of pack/permute per pass |

Their K2 shape contract does not line up with ours (`chunk_size=64`,
`[B,H,T,D]` inputs, `h` materialised for every chunk), so their kernel cannot be
dropped in, and none of this says their kernel is slower or faster than ours.
What is transferable is the *schedule*, not the code.

## 4. Measured: what the Catlass idioms are worth in our MIX kernel

Our fused one-launch-per-chunk kernel (`kernels/v1/k2_mix_all_cube.cpp`, the
`persistent_scan_cube` mode) was 3.5x slower than `separated`, and the notes
concluded that fusion does not pay.  That conclusion was an artefact: the MIX
kernel never received the micro-idioms the separated kernels had picked up
(staged once per tile, one `Fixpipe` per tile, UB `Transpose`, batched vector
updates).  Porting them one group at a time, `[1, 8192, 32, 128]`, all versions
numerically identical to the reference (`out 3.05e-05 / state 2.72e-04`, same as
`separated`):

| `kda_k2_mix_all_cube` version | total pass | K2 share |
|---|---:|---:|
| as committed (per-fractal `Fixpipe`, `Nd2Nz` operands, scalar AIV half) | 78.88 ms | ~70 ms |
| + one `Fixpipe` per 16x128 tile (`srcStride = 16`) | 60.61 ms | ~52 ms |
| + burst `DataCopy` for the 16-column operands (no `Nd2Nz`) | 42.27 ms | ~34 ms |
| + single fused d34 AIC stage (stage once, 10 Mmads, one drain) | 39.53 ms | ~31 ms |
| + UB `vtranspose` for `v_new^T`, two-repeat `Mul` for the state update | 24.75 ms | ~16 ms |
| `separated` for reference | 22.67 ms | 13.95 ms |

The same two modes at the other end of the size range, `[2, 4096, 8, 128]`
(256 chunks, 16 heads, 256 MIX launches): `separated` 6.81 ms,
`persistent_scan_cube` 6.96 ms - the fused path is 2% behind there instead of
the 3.5x it was before the port.

The five changes are 3.2x on the mode, and the mode is now within 9% of
`separated` instead of 3.5x behind.  Two negative results from the same pass:

- Removing the two `PipeBarrier<PIPE_ALL>` drains in the d12 stage
  (`run_d12_aic`, the ones whose comment warns that L0A/L0B/L0C are reused for
  `Qg`) faults with an aicore exception on this runtime; they are load-bearing.
  The equivalent drains in d3/d4 disappeared with the fused d34 stage without
  harm.  Kept.
- Fusion still does not win: 512 launches of a kernel that serialises
  AIC -> AIV -> AIC -> AIV per chunk is not faster than 2048 launches of fully
  parallel per-stage grids.  `mix_aic_1_2` (36.25 ms) and `mix_d12_vnew`
  (74.58 ms) are unchanged and still carry per-fractal `Fixpipe` calls in
  `k2_mix_d4_outstate.cpp` / `k2_mix_d12_vnew.cpp`.

Interpretation: the 2048-launch chain is *not* dominated by launch overhead.
Per chunk the separated path spends 27 us and the fused path 32 us, while the
matmul work is a few microseconds.  The cost sits in per-stage fixed costs
(load latency, pipe drains, `Fixpipe`), and in the fact that no stage can start
before the previous stage's grid has drained.  Cutting the launch count alone
does not remove either.

## 5. What to borrow, ranked

1. **One device-side chunk loop with the AIC/AIV software pipeline above**
   (their whole design).  This is the only idea that removes both the per-stage
   fixed costs and the stage-to-stage drain.  Concretely for our K2:
   `kda_k2_d12`/`vnew`/`d34`/`outstate` become four stages of one
   `KERNEL_TYPE_MIX_AIC_1_2` kernel that walks `chunk = 0..NT-1` on device,
   keeps `S32`/`S16` in per-core ping-pong workspace, and overlaps cube work of
   chunk `i+1` with vector work of chunk `i`.  Our per-chunk state is a
   64x128 tile per `(b, h, v-tile)` - small enough to keep resident - and the
   fused d34 stage built here is already the AIC half of it.
2. **Interleave two independent tasks per core** (`headInnerLoop = 2`).  With
   `BH * NV = 64` tasks on ~24 cores the chunk chain is latency-bound; a second
   chain per core is what hides it.
3. **Pre-set flags for the pipeline prologue and keep the protocol
   iteration-invariant.**  No `if (chunk > 0)` waits, one set/one wait per stage
   per iteration, flag ids <= 7, and `CrossCoreFlagWithReverse<16>` if any stage
   can run ahead.
4. **Per-core private ping-pong work buffers** instead of one global per-chunk
   buffer, so cores never couple through GM.
5. **Micro-idioms** (all measured above, and all already present in our
   separated kernels): one `Fixpipe` per tile with `srcStride = 16`; burst
   `DataCopy` for 16-column operands instead of `Nd2Nz`; UB `Transpose` for
   16x16 blocks; two-repeat `Mul` for the state update.

## 6. What not to borrow

- Their layout handling: five `transpose(1,2).contiguous()` on the inputs and
  three on the outputs, plus bf16/fp32 conversions, per call.  Our packed
  chunk-major buffers cost 0.39 ms for the whole pass.
- Their Catlass `tla` scaffolding (`PackedTileCopyTla`, 128x128x128 L1 tiles,
  `MmadPingpongTlaMulti`): it is a large dependency for kernels whose real
  problem is scheduling, and it forces the transpose contract above.
- Their variable-length plumbing (`cu_seqlens`, `chunk_indices`, dummy-head
  swizzle, context-parallel `chunk_delta_hupdate`): we do not need it.
- Materialising every chunk's `h`: only their separate `fwd_o` needs it; our
  fused `outstate` does not.
