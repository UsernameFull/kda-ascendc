# AscendC S14 MIX_AIC_1_2 Result (2026-09-08)

This stage adds an opt-in k2_mode="mix_aic_1_2" on top of the S13 full-d4 path. The default separated path and the production directory are unchanged.

## Implementation

- Added kernels/v1/k2_mix_d4_outstate.cpp.
- One MIX 1:2 logical block uses one AIC to compute two BV=64 d3 tiles and the full d4 matrix. Two AIV subblocks compute output/state.
- AIC to AIV synchronization uses CrossCoreSetFlag/CrossCoreWaitFlag, flag 9, within the logical block. It does not use a grid-wide SyncAll, so BH=32 does not block on non-resident blocks.
- AIV folds the expanded block index with GetBlockIdx()/GetTaskRation(); GetSubBlockIdx() selects the value tile.
- The d3 Cube launch is fused into the MIX kernel.
- MIX reuses d4 storage per head: (BH,128,128), overwriting the current chunk. The full-d4 comparison keeps (C,128,128).
- The new mode is explicit. separated, cube_separated, cube_d3_separated, cube_full_d4, and persistent remain available.

## Correctness

Command:

    python3 tests/test_mix_aic_1_2.py

Results:

- T=16,H=2: MIX vs full-d4 output/state max error 0.
- T=64,H=2: MIX vs full-d4 output/state max error 0.
- T=16,H=32: MIX vs full-d4 output/state max error 0.
- d4, output, and state are finite.
- H=32 completes with local cross-core flags. The earlier grid-wide SyncAll version blocked at BH=32 and was removed.

## Performance

Hardware: Ascend910_9382. Warmup=2, reps=3, device synchronization before and after each sample. Inputs, scale, normalization, and Triton settings match the S13 benchmark.

| Shape | cube_separated | cube_full_d4 | MIX 1:2 | Triton Ascend | MIX/full |
|---|---:|---:|---:|---:|---:|
| [1,32,2,128] | 3.270 ms | 0.922 ms | 0.923 ms | 0.258 ms | 1.00x |
| [2,1024,4,128] | 95.029 ms | 19.994 ms | 20.137 ms | 1.140 ms | 0.99x |
| [2,4096,8,128] | 383.578 ms | 87.764 ms | 87.156 ms | 8.215 ms | 1.01x |
| [1,8192,32,128] | 1529.447 ms | 343.398 ms | 343.526 ms | 30.082 ms | 1.00x |

MIX is 3.4x to 4.7x faster than cube_separated. It is still about 6.4x to 17.1x slower than Triton Ascend. MIX vs full-d4 numerical error is zero. MIX vs Triton error is output 0.00458 to 0.01178 and state 1.27e-4 to 4.24e-4.

For [1,8192,32,128], full-d4 d4 storage is about 1 GiB; MIX storage is about 2 MiB, a 512x reduction. End-to-end time is essentially unchanged. The remaining bottleneck is d12/vnew, K1, per-chunk launches, and GM traffic; d3/d4/output fusion alone does not approach Triton.

## Next steps

1. Build a real chunk pipeline across d12, vnew, d3, d4, and state/output to avoid AIC waiting for GM reads.
2. Keep state on-chip or in shared workspace to reduce d4 and state GM traffic.
3. Move normalization, preprocess, and K1 solve/w/u to device code and use one end-to-end timing contract.
4. Keep MIX as an experimental comparison path until device-side pipeline work shows a stable gain.
