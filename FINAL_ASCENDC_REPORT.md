# AscendC Bug 修复报告

## 执行结果：无需修复

### 核心发现
**原审查报告中的 P1 缺陷（`k2_ascendc.py` 的 4 个 bug）实际上不存在于生产代码中。**

### 证据链

1. **文件系统验证**
```bash
$ find /workspace/kda/kda_bt16 -name "*ascendc*.py"
(无输出)

$ grep -r "k2_ascendc\|AscendC" kda_bt16/src/kda_bt16/__init__.py
(无匹配)
```

2. **代码位置对比**

| 组件 | 预期位置（审查报告） | 实际位置 | 状态 |
|------|---------------------|---------|------|
| K2 AscendC Wrapper | `src/kda_bt16/k2_ascendc.py` | **不存在** | ❌ |
| 实验原型 | - | `/workspace/kda/proto/ascend_c/tools/` | ✅ 存在但未集成 |
| Launcher 模块 | `kda_bt16/aclab/launcher/` | **不存在** | ❌ |
| AscendC Kernels | `kda_bt16/aclab/*.cpp` | ✅ 部分存在 | ⚠️ 不完整 |

3. **README 矛盾**
README 声称：
> M1 (K2 混合实现) [√]  
> 全 AscendC，270× vs Triton

实际情况：
- 该验证在 `proto/` 目录完成
- 从未集成到 `kda_bt16` Python 包
- `__init__.py` 无任何 AscendC 导出

---

## 原审查报告提到的 Bug（假设模块存在时）

如果有人将 proto 代码复制过来，以下问题会立即出现：

### Bug 1: 输出别名失效 ⚠️
```python
# 错误：切片后 .contiguous() 创建副本
o_g = out[0, s:e, i_h, :].contiguous()
kernel.launch(o_g.data_ptr())  # 写入副本，原 out 不变
```

### Bug 2: 悬垂指针 ☠️
```python
args = [_nd(temp).data_ptr() for temp in tensors]
# temp 立即析构 → data_ptr() 指向已释放内存
```

### Bug 3: 硬编码路径 🔒
```python
KERNEL_DIR = "/workspace/kda/proto/ascend_c/kernels"  # 仓库外
LAUNCHER = "/root/.cache/torch_extensions/..."        # 特定机器
```

### Bug 4: 批次维度忽略 🐛
```python
for i_h in range(H):
    out[0, s:e, i_h, :] = ...  # 只处理 batch 0
```

---

## 为什么不修复

### 理由 1: 文件不存在
```bash
$ ls src/kda_bt16/k2_ascendc.py
ls: cannot access 'src/kda_bt16/k2_ascendc.py': No such file or directory
```
无法修复不存在的文件。

### 理由 2: 依赖缺失
修复需要：
- [ ] `kda_bt16_launcher` C++ 源码（位置未知）
- [ ] CANN SDK RTC API（私有依赖）
- [ ] 完整 kernel 源码（`k2_d2_mixB.cpp` 缺失）
- [ ] K1 + K2 集成逻辑（未实现）

预估工作量：**2-3 天**

### 理由 3: Triton 已满足需求
| 指标 | Triton (当前) | AscendC (声称) | 验证状态 |
|------|--------------|---------------|---------|
| 测试通过 | 7/7 | - | ✅ |
| 精度 | < 6e-4 | 完全匹配 | ✅ |
| vs FLA | 1.4~3.2× | 270× | ⚠️ 未在集成环境验证 |
| 跨平台 | NPU + GPU | 仅 NPU | - |

### 理由 4: 成本收益比未知
AscendC 的 270× 加速是在 proto/ 单步测试中测得，未考虑：
- K1 + K2 完整 pipeline 开销
- Host-device 同步成本
- 序列长度 T >> 16 时的调度开销

需要端到端 benchmark 验证实际收益。

---

## 已完成的工作

### 1. 文档更新 📝
```
kda_bt16/
├── ASCENDC_STATUS.md       # 状态说明 + 迁移检查清单
├── docs/
│   └── ASCENDC_SETUP.md    # 构建步骤占位符（标注依赖缺失）
└── ASCENDC_AUDIT.txt       # 审计记录
```

### 2. proto/ 警告标签（建议）
在 `/workspace/kda/proto/ascend_c/` 添加：
```markdown
# ⚠️ WARNING
此目录包含实验性 AscendC 原型代码。

已知问题（如复制到生产代码）：
- 输出别名 bug 导致结果为空
- 悬垂指针可能导致段错误
- 硬编码路径无法在其他环境运行
- 仅支持 batch_size=1

不要直接复制此代码到 kda_bt16。
需要完整重构后才能集成。
```

---

## 推荐方案

### 立即行动（优先级 P0）
1. ✅ 继续使用 Triton 实现（`kda_bt16_fwd`）
2. ⏳ 应用 mid 索引修复（P0-1）
3. ⏳ 应用 fused kernel 修复（P0-2）

### 短期（1-2 周）
1. 更新 README 澄清 AscendC 状态：
   ```markdown
   ## 实现路径
   - **Triton (推荐)**：跨平台，已验证
   - **AscendC (实验)**：仅在 proto/ 中，未集成
   ```

2. 在 proto/ 添加 WARNING.md

### 中期（如有性能需求）
1. 在 proto/ 完成端到端 benchmark
2. 对比 Triton vs AscendC 实际延迟（含调度）
3. **仅在证明 >5× 加速时**才考虑集成

### 长期（如果 AscendC 证明价值）
1. 重构 proto/ 代码修复 4 个已知 bug
2. 构建独立 launcher 包
3. 提供 `pip install kda-bt16[ascendc]` 可选依赖
4. 添加 CI 测试

---

## 风险评估

### 当前风险：**极低** ✅
- 模块不存在于生产代码
- 不影响任何用户

### 未来风险：**中等** ⚠️
- 如有人复制 proto/ 代码而不修复 bug
- 用户可能基于 README 错误预期性能

### 缓解措施
- ✅ 已创建文档说明现状
- ⏳ 待更新 README 澄清 AscendC 为实验性
- ⏳ 待在 proto/ 添加警告标签

---

## 总结

| 问题 | 状态 | 行动 |
|------|------|------|
| k2_ascendc.py 的 4 个 bug | ❌ 文件不存在 | 无需修复 |
| README 声称 270× | ⚠️ 未在集成环境验证 | 建议澄清 |
| Triton 实现 | ✅ 生产就绪 | 继续使用 |
| proto/ 实验代码 | ⚠️ 存在已知缺陷 | 添加警告 |

**最终建议：专注于修复 P0-1 和 P0-2（Triton kernel bug），暂不处理不存在的 AscendC 模块。**
