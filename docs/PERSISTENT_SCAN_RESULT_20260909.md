# Persistent Scan Implementation Results

Date: 2026-09-09

## Implementation

Added opt-in `k2_mode="persistent_scan"` without changing existing modes. The new RTC source is `kernels/v1/k2_persistent_scan.cpp`; the API compiles and launches it as `kda_k2_persistent_scan_kernel`.

The kernel performs the K2 chunk loop on device, keeps the FP32 state tile resident across chunks, and writes final state once at the end. It preserves the existing persistent numerical path and maps independent batch/head/value tasks to separate blocks.

## Correctness

Server tests: `pytest -q tests/test_persistent_scan.py -s`

Result: 3 passed in 44.00s.

Validated cases include `T=16,H=2`, `T=64,H=2`, and `T=16,H=32`. An additional `B=2,T=1024,H=4` comparison also passed. Output and final-state maximum absolute errors against `persistent` were both `0.0`; all results were finite.

## Performance

Synchronized wall-clock medians on Ascend910_9382:

| Shape | persistent | persistent_scan | Speedup | K2 launches |
|---|---:|---:|---:|---:|
| `[2,1024,4,128]` | `235.229 ms` | `235.144 ms` | `1.0004x` | `1` |
| `[2,4096,8,128]` | `942.011 ms` | `941.931 ms` | `1.0001x` | `1` |

Both paths use one K2 kernel launch already, so this version proves the device-side cross-chunk recurrence and isolates the mode, but does not materially reduce runtime relative to the existing persistent implementation.

## Next Optimization

First Vector pass completed on 2026-09-09: `v_new` subtraction, output scale/add, and state decay/add now use AscendC Vector primitives. Server medians improved from `235.260` to `220.250 ms` for `[2,1024,4,128]` and from `942.077` to `882.340 ms` for `[2,4096,8,128]`, about `1.068x`; output error was at most `7.63e-6` and state error at most `3.73e-9`. The remaining matrix products still use scalar `GetValue`/`SetValue` loops. The next step is Cube MMAD for `W @ state`, `Q @ state`, `A @ v_new`, and `v_new @ K`, followed by double buffering and explicit MTE/VEC/Cube overlap.


## Cube Probe

The existing isolated d12 Cube microkernel was measured on the server at `0.0429 ms` versus `2.2482 ms` for the scalar AIV implementation, or `52.4x` for that microkernel. Its numerical check passed with maximum absolute errors below `3e-8`.

This result is evidence that Cube MMAD is worthwhile, but it is not an end-to-end estimate. Integrating it into persistent scan requires a mixed AIC/AIV kernel: AIC computes both value tiles of `W/Q @ state`, AIV keeps the FP32 state recurrence, and two per-value-tile cross-core flags are needed so AIC never reads a partially updated state. The state must also use an explicit BF16 ping-pong cache for Cube input while retaining FP32 accumulation on AIV. This mixed design is the next implementation step; the current `persistent_scan` remains the safe Vector-optimized path.

## Cube Target Path

On 2026-09-09, `k2_mode="persistent_scan_cube"` was connected to the verified `cube_full_d4` execution path. This path keeps state and intermediates on NPU, uses Cube MMAD for d12/d3/d4, and performs state updates on device for each chunk. The experimental single-kernel AIC/AIV loop remains RTC-compilable, but is not used by the public mode because its cross-core loop barriers stall on the current CANN runtime.

Synchronized wall-clock medians on Ascend910_9382:

| Shape | Vector persistent scan | Cube target path | Speedup | Output error | State error |
|---|---:|---:|---:|---:|---:|
| `[2,1024,4,128]` | `219.878 ms` | `20.024 ms` | `10.981x` | `3.05e-5` | `3.41e-6` |
| `[2,4096,8,128]` | `881.699 ms` | `87.955 ms` | `10.024x` | `6.10e-5` | `2.69e-6` |
| `[1,8192,32,128]` | `3525.322 ms` | `343.914 ms` | `10.251x` | `3.05e-5` | `3.01e-6` |

All Cube target outputs were finite. The target path currently launches Cube stages once per chunk, so the remaining optimization is launch fusion or runtime-supported graph capture; the measured end-to-end Cube speedup is already above 10x versus the Vector persistent scan.

## Triton Comparison

Using the same NPU-resident inputs, synchronized wall-clock timing, two warmups, and five measured samples:

| Shape | Cube target path | Triton | Triton speedup | Cube output error | Cube state error |
|---|---:|---:|---:|---:|---:|
| `[2,1024,4,128]` | `20.069 ms` | `1.138 ms` | `17.64x` | `7.996e-3` | `2.57e-4` |
| `[2,4096,8,128]` | `87.985 ms` | `8.165 ms` | `10.78x` | `8.423e-3` | `4.83e-4` |

The Cube path is substantially faster than the Vector persistent path, but remains slower than Triton because it launches five K2-related stages per chunk (`d12`, `vnew`, `d3`, `d4`, and out/state). The next performance target is launch fusion or a runtime-supported graph capture that preserves the Cube math while removing this per-chunk host launch overhead.

## 2026-09-09 Vector Fusion Follow-up

The fused Cube path was updated to replace scalar AIV `GetValue`/`SetValue` loops in `vnew`, output accumulation, and state decay/update with Vector `Sub`, `Muls`, `Add`, and `Mul` operations. RTC compilation and the Cube regression test passed on the server.

Synchronized end-to-end medians using the existing AscendC-vs-Triton benchmark:

| Shape | AscendC after Vector fusion | Triton | Triton speedup |
|---|---:|---:|---:|
| [2,1024,4,128] | 5.414 ms | 1.132 ms | 4.78x |
| [2,4096,8,128] | 28.017 ms | 8.206 ms | 3.41x |
| [1,8192,32,128] | 109.120 ms | 30.014 ms | 3.64x |

Output and state errors remained finite and unchanged at approximately 8e-3 and 2.6e-4 for the first shape. A two-chunk AIC/AIV persistent experiment was tested with multiple flag layouts and timed out on T=64; it is not connected to the public API. The safe public route remains the one-chunk `mix_all_cube` path.
