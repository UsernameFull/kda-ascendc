# KDA on Ascend (v1) — Ascend C implementation of the Kimi Delta Attention forward

Runtime-compiled (RTC) Ascend C kernels for the KDA chunked forward pass on the
Ascend 910B, plus the Python runtime that compiles and launches them.

This is the release tree: the latest (v1) implementation only. Three device
stages, chunk-generic (`KDA_CHUNK` = 16/32/64, production build 64), bf16
inputs, `K = V = 128`, inference forward only.

## Performance

Canonical shape `[B=1, T=8192, H=96, D=128]`, Ascend 910B3 (20 AIC / 40 AIV),
`KDA_CHUNK=64`, single process:

| metric | value |
|---|---:|
| e2e (`do_bench` median) | **8.79–8.81 ms** (p20 8.785 / p80 8.805) |
| stage 1 `pre_gram` | ~3.88 ms |
| stage 2 `solve` (sliced) | ~1.92 ms (wide AIV half ~1.80) |
| stage 3 `K2` (persistent loop) | ~3.08 ms |

Stage numbers are device spans (the `KDA_PROFILE` breakdown is host-sync and is
not stage pricing). Reference points on other hardware: FLA Triton `chunk_kda`
2.722 ms on H100 CI; FlashKDA (CUTLASS, K1/K2) 1.50 ms on H800.

Chunk sizes: C=16/C=32 builds are supported, but C=64 is the production build —
at the same shape the C=16 build measures 14.95 ms e2e (K2 takes four times the
chunk steps and its per-chunk cost does not fall with the tile).

## Requirements

- Ascend 910B (the K2 loop uses `MIX_AIC_1_2`); CANN 9.1
  (`ASCEND_HOME_PATH`, default `/usr/local/Ascend/cann-9.1.0`)
- Python >= 3.10, `torch` + `torch_npu`
- `pybind11` (launcher build only), `triton` (bench script only)

## Build

```bash
# 1. Launcher: aclrtLaunchKernel wrapper + the RTC kernel compiler.
LAUNCHER_NAME=kda_ascendc_v1_launcher bash aclab/launcher/build_launcher.sh
#   -> build/S02_clean/torch_extensions/kda_ascendc_v1_launcher

# 2. Optional: precompile every kernel before first use (prints RTC_ALL_OK).
python3 tools/compile_all_server.py
```

Kernels are compiled at first call if step 2 is skipped (a few minutes).

## Run

```python
import sys
sys.path.insert(0, "python")
from kda_ascendc_v1 import kda_bt16_fwd_ascendc

# q/k/v: bf16 [B,T,H,128]; g: fp32 [B,T,H,128]; beta: fp32 [B,T,H];
# A_log: fp32 [H]; bias (dt_bias): fp32 [H,128].
out, final_state = kda_bt16_fwd_ascendc(
    q, k, v, g, beta,
    A_log=A_log, bias=dt_bias, lower_bound=-5.0,
    output_final_state=True,
)
```

The gate is the A_log + dt_bias parameterisation: `g_act = lower_bound *
sigmoid(exp(A_log) * (g + dt_bias))`, applied in-kernel together with the
q/k L2 norms and the beta sigmoid. `initial_state` is fp32 `[B,H,128,128]`.

Benchmark (the same entry used for the table above):

```bash
ASCEND_RT_VISIBLE_DEVICES=0 KDA_CHUNK=64 python3 -u tools/bench_baseline.py --tag release
```

It prints the e2e median and the per-stage profile and writes
`results/bench/baseline_<tag>.json`.

## Design

- **Stage 1 — `pre_gram`** (`kda_pre_gram_mix`): L2 norms, beta sigmoid, gate
  activation + chunk cumsum, decay application, and both intra-chunk Grams in
  one MIX block per `2 x unroll` chunks. The normalized q/k and the chunk gate
  never round-trip to GM (they were the bulk of the old preprocess traffic).
- **Stage 2 — `solve`** (`kda_solve_wu_wide` + `kda_solve_assemble` +
  `kda_solve_wu_cube`): at C >= 64 the forward substitution is two-level
  (SB=2): the AIV half solves the two 32x32 diagonal sub-blocks of every
  chunk, the AIC half forms the coupling block `X21 = -X22 L21 X11` and then
  produces `W`/`U` on the Cube. The AIV-only and AIC-only halves are launched
  on two streams as whole-AIV-wave slices.
- **Stage 3 — `K2`** (`kda_k2_persistent_loop`): one MIX launch runs the whole
  chunk recurrence; the fp32 state stays in UB for the entire sequence and
  only its bf16 copy is published to GM for the Cube operands; the output is
  written straight into the caller's `[B,T,H,D]` tensor.
- **Chunk-generic**: `KDA_CHUNK` selects the chunk size at RTC compile time
  and every stage (including the K2 loop) follows it.

Configuration knobs (env, read per build/call): `KDA_CHUNK`,
`KDA_SOLVE_WIDE_NCHUNK` (wide-solve chunks per tile, default 12 at C=64),
`KDA_PERSIST_LOOP_MAXH` (heads per K2 block, default 4), `KDA_PROFILE=1`
(per-stage profile, read back with `kda_ascendc_v1.get_last_profile()`).

## Layout

```
README.md
LICENSE
aclab/launcher/build_launcher.sh   # builds the torch-extension launcher
aclab/launcher/launcher.cpp        # aclrtLaunchKernel + rtc_compile (pybind11)
kernels/v1/*.cpp                   # Ascend C kernel sources (RTC-compiled)
python/kda_ascendc_v1/
    api.py                         # compile table, launch pipeline, public entry
    layout.py                      # pack/unpack helpers
    experimental.py                # historical C=16-only K2 modes (oracles)
    __init__.py
tools/bench_baseline.py            # single-process e2e + stage bench
tools/compile_all_server.py        # precompile every kernel
```

## Limitations

- Inference forward only; no backward/training path.
- bf16 `q/k/v`, fp32 `g`/`beta`; `K = V = 128`; `T % KDA_CHUNK == 0`; MHA
  (`HV == H`); single device (no context parallel).
- Ascend 910B only (`MIX_AIC_1_2`, Ascend C RTC).
- The per-chunk K2 modes in `experimental.py` are C=16-only historical
  kernels; the public entry serves the chunk-generic `persistent_loop` only.
