# AscendC S15 d12/vnew MIX Result (2026-09-08)

S15 adds an opt-in `k2_mode="mix_d12_vnew"`. It fuses the d1/d2 Cube work and
the AIV `vnew/vnewT` transform into one `KERNEL_TYPE_MIX_AIC_1_2` launch per
chunk. The validated S14 d3/d4/output/state kernel remains unchanged.

## Correctness

- S15 source: `kernels/v1/k2_mix_d12_vnew.cpp`.
- The standalone `k2_d12_cube.cpp` received the same d1-to-d2 `PIPE_ALL`
  barrier. Without it, multi-block H=32 runs had nondeterministic L0 reuse
  errors.
- S15 and `cube_full_d4` matched exactly for output, state, d1, d2, vnew,
  vnewT, d3 and d4 on T=16/H=2, T=64/H=2 and T=16/H=32.
- All tested tensors were finite and H=32 completed without deadlock.
- The kernel allocates a full FP32 U tile for the `M*D` cast before slicing the
  value tile; this avoids the undersized temporary used by the old standalone
  vnew implementation.

## Performance

Ascend910_9382, warmup=2, reps=3, synchronized before and after each sample.
The same normalized inputs, initial state, scale and Triton settings were used
for every mode.

| Shape | cube_full_d4 | S14 MIX | S15 d12/vnew | Triton Ascend | S15 / Triton |
|---|---:|---:|---:|---:|---:|
| [1,32,2,128] | 0.931 ms | 0.930 ms | 0.925 ms | 0.253 ms | 3.66x |
| [2,1024,4,128] | 20.018 ms | 20.135 ms | 20.484 ms | 1.140 ms | 17.97x |
| [2,4096,8,128] | 94.020 ms | 91.898 ms | 91.066 ms | 8.295 ms | 10.98x |
| [1,8192,32,128] | 343.990 ms | 344.305 ms | 348.001 ms | 30.140 ms | 11.55x |

Relative to S14 MIX, S15 ranges from 1.0% faster to 1.7% slower. The launch
reduction is therefore not sufficient to offset the extra local synchronization
and repeated AIC/AIV setup. S14 `mix_aic_1_2` remains the better experimental
K2 path; Triton Ascend remains the recommended performance path.

## Next optimization

The next material gain must remove the per-chunk Python launch loop and keep
state on device across chunks. The high-value sequence is a persistent K2 loop
with double-buffered d12/vnew and d3/d4, followed by device-side K1
normalization, gate, beta and solve. Further d12/vnew launch fusion alone is
not a competitive path.
