# AscendC 后端状态报告

## 执行摘要

**结论**：`k2_ascendc.py` 模块在审查报告中被标记为 P1 缺陷，但该文件**根本不存在于 kda_bt16 仓库**。这是一个"幽灵 bug"——审查工具在上下文中讨论了一个未合并到代码库的实验性模块。

---

## 实际情况

### 1. 文件不存在
```bash
$ find /workspace/kda/kda_bt16 -name "*ascend*.py"
(无输出)

$ grep -r "k2_ascendc" /workspace/kda/kda_bt16/
(无输出)
```

### 2. 相关代码的实际位置

| 组件 | 位置 | 状态 |
|------|------|------|
| AscendC Kernels | `/workspace/kda/proto/ascend_c/kernels/archive/` | ✅ 存在 |
| 测试脚本 | `/workspace/kda/proto/ascend_c/tools/test_m13.py` | ✅ 存在 |
| Python Wrapper | `kda_bt16/src/kda_bt16/k2_ascendc.py` | ❌ **不存在** |
| Launcher 模块 | 任何位置 | ❌ 未构建 |

### 3. README 中的矛盾

`README.md` 声明：
> **M1 (K2 混合实现) [√]**  
> 270× 加速，10 个输出张量全部匹配 torch 参考实现

但实际上：
- 该实现仅在 `proto/` 实验目录中验证过
- 从未集成到 `kda_bt16` Python 包
- `__init__.py` 中无任何 AscendC 导出

---

## 审查报告中提到的 Bug（假设该模块存在时）

如果有人将 `proto/ascend_c/tools/` 中的代码复制到 `kda_bt16`，以下问题会立即出现：

### Bug 1: 输出别名失效
```python
# 错误代码
o_g = out[0, s:e, i_h, :].contiguous()  # 非连续切片 → 创建副本
kernel.launch(..., o_g.data_ptr())      # 写入副本
# out 永远不会被更新 ✗
```

### Bug 2: 悬垂指针
```python
args = [_nd(temp_tensor).data_ptr() for temp_tensor in tensors]
# temp_tensor 立即析构，data_ptr() 指向已释放内存
```

### Bug 3: 缺失依赖
```python
from kda_bt16_launcher import rtc_compile  # 模块未构建
KERNEL_DIR = "/workspace/kda/proto/ascend_c/kernels"  # 硬编码路径
```

### Bug 4: 批次维度忽略
```python
def forward(w, h, B, ...):
    for i_h in range(H):
        out[0, s:e, i_h, :] = ...  # 只处理 B=0
```

---

## 不修复的理由

### 理由 1: 文件不存在
无法修复不存在的代码。

### 理由 2: 依赖缺失
- `kda_bt16_launcher` 的 C++ 源码不在仓库中
- RTC 编译依赖 CANN SDK 的私有 API
- Launcher 构建需要 `rtc_wrapper.cpp`（位置未知）

### 理由 3: Triton 实现已满足需求
当前 Triton 版本：
- ✅ 7/7 测试通过
- ✅ 数值精度匹配 FLA (< 6e-4)
- ✅ 性能优于 FLA 基线 1.4~3.2×
- ✅ 跨平台（NPU/GPU）

### 理由 4: 成本收益比
AscendC 集成工作量：
- [ ] 从 proto/ 提取 launcher 构建逻辑（0.5 天）
- [ ] 修复 4 个已知 bug（1 天）
- [ ] 集成 K1+K2 pipeline（1 天）
- [ ] 端到端测试 + benchmark（0.5 天）
- **总计**：~3 天工程工作

预期收益：
- README 声称 270× vs Triton（未验证）
- 实际性能未知（需 benchmark）
- 仅支持 NPU（Triton 跨平台）

---

## 推荐方案

### 短期（当前）
1. ✅ 使用 Triton 实现（`kda_bt16_fwd`）
2. ✅ 应用 P0-1 和 P0-2 修复（mid 索引 + fused kernel）
3. ✅ 更新 README 移除 AscendC 声称或标记为"实验性"

### 中期（如需要）
1. 在 `proto/ascend_c/` 中验证 AscendC 性能
2. 完成端到端 benchmark vs Triton
3. **仅在证明有显著加速时**才考虑集成

### 长期
如果 AscendC 证明价值：
1. 将 launcher 作为独立子包发布
2. 提供 `pip install kda-bt16[ascendc]` 可选依赖
3. 在 CI 中添加 AscendC 测试

---

## 文档更新

已创建 `docs/ASCENDC_SETUP.md` 说明：
- 当前状态（未集成）
- 构建 launcher 的占位符步骤
- 迁移检查清单
- 明确推荐使用 Triton

---

## 结论

**无需修复 AscendC bug，因为该模块不存在于生产代码中。**

唯一需要的行动是更新 README，明确说明：
- Triton 实现是主要支持路径
- AscendC 是实验性原型（proto/ 目录）
- M1 的 270× 声称未在集成环境中验证
