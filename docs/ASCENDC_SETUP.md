# AscendC 后端设置与使用

## ❌ 当前状态
`k2_ascendc.py` 模块**不存在于此仓库**。之前的代码审查中提到的 P1 bug 是指如果该模块存在时会有的问题，但实际上：
- 原始 `k2_ascendc.py` 从未合并到 `kda_bt16` 包
- 相关实验代码在 `/workspace/kda/proto/ascend_c/tools/`
- Launcher 模块需要从源码构建

## 构建 Launcher（前置依赖）

```bash
cd /workspace/kda/proto/ascend_c
# 创建 launcher 构建脚本
cat > build_launcher.py << 'PYTHON'
from torch.utils.cpp_extension import load
import os

launcher = load(
    name='kda_bt16_launcher',
    sources=['launcher/rtc_wrapper.cpp'],  # 假设源文件
    extra_cflags=['-O3'],
    extra_ldflags=['-lacl_op_compiler', '-lascendcl'],
    build_directory='/root/.cache/torch_extensions/py312_cpu/kda_bt16_launcher',
    verbose=True
)
print(f"✓ Launcher built at: {launcher.__file__}")
PYTHON

python build_launcher.py
```

**问题**：launcher 源码 (`rtc_wrapper.cpp`) 不在此仓库中，需要从以下位置获取：
1. CANN SDK 示例
2. 或使用 proto/ 中已编译的版本（如果存在）

## 验证 Archive Kernels

Proto 中的实现在 `kernels/archive/`：

```bash
cd /workspace/kda/proto/ascend_c
ls -1 kernels/archive/*.cpp
# k2_m1.cpp           - 双矩阵乘 (w@h^T, qg@h^T)
# kda_bt16_smoke.cpp  - Aqk @ v_new
# kda_k2_m128.cpp     - v_new^T @ kg
```

运行单步测试（需要先构建 launcher）：
```bash
cd /workspace/kda/proto/ascend_c/tools
ASCEND_RT_VISIBLE_DEVICES=1 python test_m13.py
```

## 迁移到生产包（待完成）

完整迁移需要：
1. ✅ 修复输出别名 bug（已在 `k2_ascendc_fixed.py` 中解决）
2. ✅ 支持多 batch（架构已准备）
3. ❌ 集成 K1 + K2 完整 pipeline
4. ❌ 构建 launcher 到 `kda_bt16/aclab/`
5. ❌ 性能 benchmark vs Triton

## 当前推荐：使用 Triton 实现

鉴于：
- Triton 版本已验证（7/7 tests passed）
- 性能优于 FLA 基线 1.4~3.2×
- AscendC 需要额外 2-3 天工程工作

**建议先使用 Triton 进行生产部署**。

## AscendC 性能预期

根据 README 中的声明（未验证）：
- 单步延迟：~0.07 us (vs Triton ~19 us)
- 理论加速：270× 

⚠️ 这些数字基于早期 prototype，当前无法复现。
