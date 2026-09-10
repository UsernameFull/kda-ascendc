# Triton-Ascend environment notes

The Triton path (`src/kda_bt16/`) needs a working `triton-ascend` install. Two
independent problems show up on this box; both are handled by
`scripts/fix_triton_ascend_env.py`.

## 1. `triton` clobbered by the vanilla wheel

**Symptom** (any `import triton`, including `pytest tests/test_torch_reference.py`):

```
ImportError: cannot import name 'ascend' from 'triton._C.libtriton'
```

**Cause**: `pip install torch` / `pip install -U torch` pulls the vanilla
`triton` wheel, which overwrites `triton/_C/libtriton.so` and the Python sources
that `triton-ascend` installed. The `triton_ascend-*.dist-info` metadata stays
behind, so the mix is not obvious from `pip list`.

**Fix**: reinstall the Ascend wheel (`pip install --no-deps <triton_ascend-*.whl>`),
moving the clobbered tree to `/tmp/triton_ascend_repair_<timestamp>/` first. The
script looks for the wheel in `$TRITON_ASCEND_WHEEL`, then
`/data/*/triton-ascend-vendor/`, then the Huawei mirror.

## 2. CANN >= 9.1 enumerator rename

**Symptom** (first Triton kernel launch, while the driver builds its helper):

```
error: 'RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE' is not a member of 'rtLimitType_t'
```

**Cause**: `triton/backends/ascend/npu_utils.cpp` maps `"WARP_STACK_SIZE"` to
`RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE`. CANN 9.1 dropped that enumerator and
ships `RT_LIMIT_TYPE_SIMT_STACK_SIZE` instead
(`$ASCEND_HOME_PATH/<arch>-linux/pkg_inc/runtime/runtime/base.h`).

**Fix**: the script inserts a guarded `#define` after the last `#include` of
`npu_utils.cpp`:

```cpp
#ifndef RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE
#ifdef RT_LIMIT_TYPE_SIMT_STACK_SIZE
#define RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE RT_LIMIT_TYPE_SIMT_STACK_SIZE
#else
#define RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE RT_LIMIT_TYPE_STACK_SIZE
#endif
#endif
```

It is inert on CANN releases that still define the old name and only affects
the `set_device_limit(...)` helper, which the KDA kernels do not call. The
file is backed up next to the original as `npu_utils.cpp.bak-<timestamp>`.

## Verify

```bash
python scripts/fix_triton_ascend_env.py --check           # both checks green
python -m pytest tests/test_torch_reference.py -k npu -q  # Triton kernels on NPU
```

The script's own verification step builds the driver helper (`NPUUtils()`) and
prints the compiled `npu_utils.so` path.
