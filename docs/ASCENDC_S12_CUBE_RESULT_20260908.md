# AscendC S12 Cube d1/d2 Optimization Results

Date: 2026-09-08
Environment: Ascend910B3, CANN 9.1.0, torch_npu 2.12.0

## Implementation

Added k2_mode="cube_separated". It replaces only d1/d2 in the separated K2 path:

- d1 = W @ state^T
- d2 = Qg @ state^T

The AIC-only Cube microkernel is in kernels/v1/k2_d12_cube.cpp. vnew, d3, d4, output, and state update continue to use the validated separated AIV kernels. The existing separated and persistent modes are unchanged.

## Correctness

tests/test_d12_cube.py passed for BH=1/2/4/8. Cube microkernel max error was below 3e-8, with canary and finite checks passing.

tests/test_cube_separated.py matched the separated path:

| Shape [B,T,H,128] | output max abs | state max abs |
|---|---:|---:|
| [1,32,2,128] | 0 | 0 |
| [2,1024,4,128] | 4.77e-7 | 0 |
| [2,4096,8,128] | 3.81e-6 | 0 |
| [1,8192,32,128] | 3.81e-6 | 0 |

Persistent regression also passed: low-level output 1.22e-4 and state 3.31e-5; S07 closure and edge tests passed.

## Performance

tools/bench_s12_one.py used the same input, synchronization, warmup=1, and reps=3:

| Shape | separated | cube_separated | speedup | persistent | Triton Ascend |
|---|---:|---:|---:|---:|---:|
| [1,32,2,128] | 7.687 ms | 3.263 ms | 2.36x | 7.646 ms | 0.258 ms |
| [2,1024,4,128] | 235.568 ms | 95.070 ms | 2.48x | 234.880 ms | 1.141 ms |
| [2,4096,8,128] | 945.048 ms | 383.580 ms | 2.46x | 941.603 ms | 8.243 ms |
| [1,8192,32,128] | 3776.686 ms | 1529.584 ms | 2.47x | 3765.118 ms | 30.163 ms |

Cube remains 12.7-83.3x slower than Triton Ascend. The remaining bottleneck is AIV vnew, d3/d4, state update, and repeated kernel launches. The d1/d2 gain must not be treated as an end-to-end production backend result.

## Decision

- Keep cube_separated as an optional experimental optimization path.
- Keep persistent as the correctness baseline; it fuses d1/d2 inside one AIV kernel and has no simple Cube insertion point yet.
- Keep Triton Ascend as the default high-performance path.
- Next optimization should be one KERNEL_TYPE_MIX_AIC_1_2 pipeline for d1, AIV vnew/qg/kg, d2/d3/d4, output, and state update, reducing GM round trips and launch count.

## Reproduction

python3 tests/test_d12_cube.py
python3 tests/test_cube_separated.py
python3 tools/bench_s12_one.py
python3 tests/test_persistent.py
python3 tests/test_persistent_edges.py
python3 tests/test_s07_persistent.py
