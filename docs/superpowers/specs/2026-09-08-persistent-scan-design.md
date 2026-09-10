# Persistent Scan Design

Date: 2026-09-08
Status: Approved for implementation

## Goal

Add an opt-in `k2_mode="persistent_scan"` that moves the cross-chunk K2 state recurrence into one AscendC kernel launch, while preserving the existing `persistent`, `mix_d12_vnew`, and separated modes.

## Scope

The first implementation targets the existing fixed contract: BF16 Q/K/V, FP32 preprocessing inputs and state, `D=128`, `BT=16`, `NV=2`, and `T % 16 == 0`. The implementation must support arbitrary validated `B`, `H`, and `T` values already accepted by the API.

No Triton changes, Cube tile redesign, or numerical contract changes are included in this phase.

## Execution Model

The Python API continues to run preprocess, Gram, solve, and state initialization as separate kernels. It then launches one `kda_k2_persistent_scan_kernel` for K2.

The kernel maps one logical task to each `BH * NV` state lane. Each task loads its initial state once, loops over chunks in increasing time order, and performs the existing d12, vnew, d3/d4, output, and state-update operations before advancing to the next chunk. The state recurrence remains sequential within a task; independent batch/head/value lanes remain parallel.

The kernel writes output tiles for every chunk and writes final state once at the end. It must not use grid-wide synchronization because the existing implementation has already shown that grid-wide synchronization can block for large head counts.

## API and Compatibility

- Add `persistent_scan` to the accepted `k2_mode` values.
- Keep `persistent` unchanged as the correctness and performance baseline.
- Compile the new source lazily with the existing RTC mechanism.
- Record the new kernel in `get_last_profile()` launch counts.
- Do not change default mode selection or existing output layouts.

## Numerical Contract

Reuse the existing persistent path's FP32 state accumulation, BF16 intermediate/output layout, initial-state handling, scale, and final-state reshape. The new path is correct only when output and final-state errors against the existing `persistent` path are within the existing test tolerances and all outputs are finite.

## Validation

Run correctness tests in this order:

1. `T=16,H=2`
2. `T=64,H=2`
3. `T=16,H=32`
4. `B=2,T=1024,H=4`
5. `B=2,T=4096,H=8`

For each case compare output and final state against `persistent` and `cube_full_d4`, and record launch counts. Then run the standard synchronized median benchmark for the two longer cases. The benchmark must report regular persistent latency, persistent-scan latency, speedup, launch counts, maximum absolute errors, and finite checks.

## Failure Handling

`persistent_scan` is opt-in and must fail clearly if compilation or execution is unsupported. Existing modes must remain callable and must not silently route through the new kernel. No automatic numerical fallback is introduced in the first implementation, because hiding a broken device path would invalidate performance measurements.

## Out of Scope

- Cross-task time parallel scan.
- Triton optimization.
- Graph capture integration for the new kernel.
- Changes to the public `src/kda_bt16` Triton API.
- Changes to individual Cube tile sizes.
