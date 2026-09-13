# KDA forward at `[1, 8192, 96, 128]`: us vs FLA

Date: 2026-09-12 (measures). Device: Ascend 910B3 (`Ascend910_9382`, 24 cube /
48 vector cores, 64 GB HBM, 192 MB L2), CANN 9.1, torch_npu 2.12, triton-ascend 3.2.0.
Shape `[B, T, H, D] = [1, 8192, 96, 128]` is FLA's `B1_T8192_H96_D128`
benchmark config (`fla/benchmarks/ops/registry.py`), bf16 q/k/v, fwd only,
`lower_bound=-5`, `safe_gate=True`, `use_qk_l2norm_in_kernel=True`.

`[1, 8192, 96, 128]` 是 FLA 官方 benchmark 的六个默认 shape 之一，因此下面的
数字可以直接和 FLA 公开的 H100/H200 CI 数字对照。

## Numbers (median of `triton.testing.do_bench`, warmup 100 ms / rep 2000 ms)

| implementation | ms |
|---|---:|
| **ours, AscendC `k2_mode="persistent_loop"`** | **14.82** (3 runs: 14.84 / 14.82 / 14.82) |
| ours, AscendC `k2_mode="separated"` (2048 launches) | 40.9 |
| ours, Triton two-stage BT=16 (`kda_bt16_fwd`) | **does not run** (see below) |
| FLA `chunk_kda` chunk_size=64, triton-ascend NPU backend, FLA's own `state_v_first=False` | 69.57 |
| FLA `chunk_kda` chunk_size=32, same | 54.41 |
| FLA `chunk_kda` chunk_size=64, `state_v_first=True` (our layout) | 81.16 |
| FLA `chunk_kda` chunk_size=32, `state_v_first=True` | 77.07 |
| FLA `chunk_kda` chunk_size=64 on FLA's H100 CI runner (published, GPU reported as H200) | 2.722 |

Ratios: ours is **3.7x** faster than FLA's best Ascend config (chunk32, 54.4 ms)
and **4.7x / 5.5x** faster than chunk64 (69.6 / 81.2 ms). Against the published
H100-class number we are **5.4x slower** (14.82 vs 2.72 ms); FLA itself is
**~20-26x slower on this 910B3** than on its H100/H200 runner.

Two reliability caveats at this shape (both measured, both new):

- `persistent_loop` faults with `507015 aicore exception` on roughly 1 launch
  in 3 at `[1,8192,96,128]` (attempt 1 of 3 here; when it runs it is stable to
  +-0.3%). The repo already documents this class of fault for the big shapes
  (missing `PipeBarrier<PIPE_ALL>` hazard); this shape is on the edge.
- the *uncommitted* WIP K1 path (`KDA_PRE_GRAM=mix`, `kernels/v1/k1_pre_gram_mix.cpp`,
  now the default in `python/kda_ascendc_v1/api.py`) does not complete at this
  shape: `507014 aicore execution times out` twice, then a >500 s hang. All the
  numbers above use the original AIV path, `KDA_PRE_GRAM=aiv`.

The A_log/dt_bias parameterisation is worth <1% for FLA on this device
(V-first, chunk64: 81.01 ms with A_log vs 81.16 ms in FLA's registry config;
chunk32: 77.90 vs 77.07), so the cross-config comparison above is clean. The
layout convention is what matters: FLA's default (K-first) is 16-42% faster
than the V-first layout our kernels emit.

## Where the H100 number comes from

FLA's own benchmark CI posts a table on every PR. For this shape the published
value is 2.722 ms (`chunk_kda`, fwd, 1/8192/96/128):

- comment on `fla-org/flash-linear-attention` PR #858 (2026-04-22), job
  "NVIDIA-H100-PT2-7": `chunk_kda | fwd | 1 | 8192 | 96 | 128 | 2.725 | 2.722 | 1.00x`
- the same row appears in PR #833 (2026-04-16): 2.725 / 2.718 ms

Caveat: the workflow/runner is named `nvidia-h100-ci` (self-hosted runner
`nvidia-h100-1`) but the machine info the harness prints is
`GPU: NVIDIA H200 | CUDA 12.8 | PyTorch 2.7.1+cu128`, i.e. the H100 pool
currently reports H200 silicon. FLA's CI config (`reusable-ci-benchmarks.yml`)
is `FLA_BENCH_WARMUP_MS=40`, `FLA_BENCH_REP_MS=200`, 6 op warm-up iters, and
`FLA_DISABLE_BACKEND_DISPATCH=1` (Triton baseline only, no A_log/dt_bias).
No newer published row for this shape exists (checked PRs #833, #858, #1047,
#1052, #1109, #1121, #1144, #1221).

## Accuracy at this shape (A_log + dt_bias config, FLA chunk64 as reference)

| pair | o max-abs | o rel-L2 | ht max-abs | ht rel-L2 |
|---|---:|---:|---:|---:|
| FLA chunk32 vs chunk64 (noise floor) | 4.9e-04 | 4.9e-04 | 5.4e-04 | 5.0e-05 |
| AscendC `persistent_loop` vs chunk64 | 7.3e-04 | 4.5e-03 | 5.7e-03 | 2.3e-03 |
| AscendC `separated` vs chunk64 | 7.3e-04 | 4.4e-03 | 5.7e-03 | 2.3e-03 |

Both AscendC modes agree with each other bit-for-bit and stay inside the
repo's `out_err < 1e-3` bound; ht is ~10x the chunk-size noise floor but small
in absolute terms (the state entries are O(1-10)).

## Triton two-stage proto cannot run at H=96

`kda_bt16_fwd` (the BT=16 two-stage path) never returns at this shape: the
K1 (intra-chunk) launch at grid `NT*BH = 512*96 = 49152` does not complete
(200 s timeout; the earlier full run died with `507014 aicore execution times
out` after ~10 min). Isolated with identical harnesses:

| kernel | H | T | grid | time |
|---|---:|---:|---:|---:|
| K1 | 32 | 8192 | 16384 | 0.011 s |
| K1 | 96 | 8192 | 49152 | **never returns** |
| K1 | 96 | 1024 | 6144 | 0.12 s (compile+run) |
| K1 | 96 | 2048 | 12288 | **never returns** (200 s) |
| K2 | 48/64/96 | 8192 | 48-96 | 19-29 ms |

So the break is specific to K1 once `H == 96` and `T >= 2048`; K2 is healthy.
The AscendC path (which packs tokens and uses its own kernels) is unaffected,
so all "ours" numbers above come from `kda_bt16_fwd_ascendc`.

## Reproduce

```bash
# ours + FLA on the same NPU, same tensors, FLA's do_bench methodology
ASCEND_RT_VISIBLE_DEVICES=0 python benchmarks/bench_fla_compare.py \
    --shape 1,8192,96,128 --impl fla-bench,fla-alog,ascendc \
    --ascendc-modes persistent_loop,separated --warmup-ms 100 --rep-ms 2000
# JSON lands in results/bench/fla_compare_1_8192_96_128.json

# FLA's own harness (A2 CI form), if you want the same number from FLA's CLI
cd /tmp/fla-src2 && ASCEND_RT_VISIBLE_DEVICES=0 FLA_BENCH_WARMUP_MS=25 \
  FLA_BENCH_REP_MS=100 FLA_BENCH_OP_WARMUP_ITERS=3 python -m benchmarks.ops.run \
  --op chunk_kda --base '' --modes fwd \
  --custom-shapes '{"B1_T8192_H96_D128": {"B": 1, "T": 8192, "H": 96, "D": 128}}'
```

Environment note: FLA needs `einops` (installed 2026-09-12) and
`/tmp/fla-src2` (FLA main @ 516143e) on `sys.path`; the KDA NPU backend
(`fla/ops/kda/backends/triton_ascend`, PR #1047) is auto-dispatched on NPU and
is what "FLA triton-ascend" means here.
