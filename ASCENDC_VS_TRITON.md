# AscendC vs Triton 性能对比分析

**文档日期**: 2026-09-04  
**基准环境**: 华为 Ascend 910B3, CANN 8.0

---

## 📊 核心结论

### Triton 实现（生产可用）
✅ **已部署**: kda_bt16_fwd (两段式)  
✅ **性能**: 1.4~3.2× vs FLA baseline  
✅ **状态**: 生产就绪，11/11 测试通过

### AscendC 实现（实验阶段）
⚠️ **状态**: 原型验证，未集成  
📊 **单步延迟**: ~0.07 us (kernel 级别)  
❌ **阻塞**: 2 个编译器 bug + 无生产 wrapper

---

## 🏁 端到端性能对比

### Triton 两段式（实测）

| 配置 | Triton BT=16 | FLA chunk64 | 加速比 |
|------|--------------|-------------|--------|
| B=2, T=1024, H=4 | **1.19 ms** | 3.85 ms | **3.2×** |
| B=2, T=4096, H=8 | **5.69 ms** | 9.31 ms | **1.6×** |
| B=1, T=8192, H=32 | **20.5 ms** | 28.4 ms | **1.4×** |

**特点**:
- ✅ 端到端完整测量（含调度开销）
- ✅ 包含 kernel 启动、内存分配、同步
- ✅ 真实生产场景性能

---

### AscendC M1（单步 kernel 延迟）

**声称**: ~0.07 us/step (kernel 执行)  
**对比 Triton K2**: ~19 us/step (单个 Triton kernel)

**理论加速**: ~270× (kernel 级别)

**BUT ⚠️ 关键问题**:

```
单步延迟 ≠ 端到端性能
```

#### 未计入的开销

1. **K1 Aqk 计算**: AscendC 版本也需要
2. **Kernel 启动开销**: 
   - Triton: NT 次 kernel 启动（T=8192 时 512 次）
   - AscendC M1: 每步独立启动 → 512 次启动
3. **Host-device 同步**:
   - 每步 `torch.npu.synchronize()` → 管道中断
4. **状态传递开销**:
   - 跨 kernel 的隐状态传递

#### 实测投影（Round 51-52）

```
AscendC M1 (H=32, T=8192): ~35.5 ms (投影)
Triton 两段式:              20.3 ms (实测)

实际: Triton 快 1.75×
```

**原因**: 
- 单步 kernel 优势 (270×) 被 512 次启动开销抵消
- 每步同步破坏流水线
- 缺少多核调度

---

## 🔍 详细技术对比

### 实现架构

| 维度 | Triton 两段式 | AscendC M1 | AscendC M2 (blocked) |
|------|--------------|-----------|---------------------|
| **Kernel 数量** | 2 个 | 1 个/步 × NT | 1 个持久化 |
| **启动次数** | 2 | 512 (T=8192) | 1 |
| **状态管理** | K2 on-chip resident | 跨 kernel 传递 | 512 步循环内 |
| **并行度** | NT × BH 程序 | 单序列化 | B=8 block |
| **调度开销** | 2 次 kernel launch | 512 次 + 同步 | 1 次 |

### 代码复杂度

```
Triton 两段式:
  - kernels.py: ~500 行 Triton DSL
  - 可移植 (CUDA/NPU)
  - 自动调优、内存管理

AscendC M1:
  - 多个 .cpp kernel (d1/d2/d3/d4/mix)
  - 手动 L0/L1/GM 管理
  - 手动同步、Cube/Vector 混合编程
  - Python wrapper 被移除（见 aclab/README.md）

AscendC M2:
  - 512 步循环 + 动态控制流
  - 被 2 个编译器 bug 阻塞
```

---

## 🐛 AscendC 当前障碍

### Bug 1: 动态循环 + Cube = 错误结果

```c++
// 即使 NT=1，包在 for 里就错
for (int t = 0; t < NT; t++) {  
    Mmad(16, 128, 128);  // ❌ 结果错误 (3.58 vs 4.8e-7)
}

// 展开就对
Mmad(...);  // ✅ 正确
Mmad(...);  // ✅ 正确
```

**影响**: M2 持久化 kernel 无法实现

### Bug 2: 多 Mmad 块 → 交叉污染

```c++
// B=1 单块
Mmad(...);  // ✅ 1.2e-7 正确

// B=2 两块（完全独立的数据）
Block 0: Mmad(...);  // ❌ 2.09 错误
Block 1: Mmad(...);  // ❌ 1.75 错误
```

**影响**: 多块并行执行损坏

### 状态总结

| 实现 | 正确性 | 性能 | 生产可用 |
|------|--------|------|---------|
| M1 单步 | ✅ 10/10 输出匹配 | ⚠️ ~35.5ms 投影 | ❌ 无 wrapper |
| M2 持久化 | ❌ bug 阻塞 | 🔮 未知 | ❌ |
| Triton | ✅ 11/11 | ✅ 20.5ms | ✅ |

---

## 📈 性能分析深度

### K2 瓶颈剖析（Triton）

```
每步耗时分解:
  tl.dot 执行:    76.5% (~3-5 us 固定成本)
  内存访问:       15%
  其他:           8.5%

算力利用:        ~0.4 TFLOPS (理论 256 TFLOPS)
```

**结论**: K2 不是带宽瓶颈，是算力利用率问题

### 为什么 AscendC 单步快？

```
Cube 矩阵单元:
  - 硬件加速 16×16×16 块矩阵乘
  - 直接操作 L0 缓存
  - 避免 Triton 中间抽象层

单步 0.07 us vs Triton 19 us:
  - Triton 包含 kernel 启动、参数传递、内存分配
  - AscendC 测量的是纯计算（已在 L0）
```

### 为什么端到端反而慢？

```
AscendC M1: 512 步 × (0.07us kernel + 69us 开销) ≈ 35.5ms
            ^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            计算时间    启动+同步+状态传递

Triton:     2 kernel × (10ms 计算 + 0.25ms 启动) ≈ 20.5ms
            ^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^
            充分流水    大 grid 并行
```

---

## 🎯 路线图状态

### 已完成 ✅
- Triton 两段式生产实现
- AscendC M1 kernel 级验证

### 实验中 ⚠️
- AscendC M1 多核调度（无 wrapper）
- 块并行方案探索

### 阻塞 ❌
- AscendC M2 持久化（编译器 bug）
- 批量 matmul 优化（同一 bug）

### 已放弃 🚫
- 单 kernel 融合（1.5× 慢）
- 寄存器求解（10% 慢）
- 算子预计算（仅 -1.6%）
- 前缀扫描融合（matmul bound）

---

## 💡 关键洞察

### 1. Kernel 延迟 ≠ 端到端性能

```
理论加速 270× → 实际慢 1.75×
```

启动开销、同步、调度在序列化执行中占主导

### 2. 并行度 > 单步效率

```
Triton: NT×BH 并行程序  
AscendC: 序列化执行

→ Grid 并行胜过单步优化
```

### 3. 抽象层成本可控

```
Triton DSL 开销 < 调度灵活性收益
```

### 4. 编译器成熟度关键

```
Triton: 稳定，可移植
AscendC: 2 个 P0 bug 阻塞生产
```

---

## 📊 数值精度对比

### Triton 两段式
```
vs fp32 recurrent gold:
  o:  4.9e-4 (bit-identical to FLA)
  ht: 1.8e-3 (bit-identical to FLA)
```

### AscendC M1
```
10/10 中间张量匹配 torch 参考
精度: bf16 级别 (~1e-3)
```

**结论**: 两者精度相当

---

## 🔮 未来展望

### 短期（如编译器修复）

**M2 持久化可能达到**:
```
理论: 512 步 × 0.07us ≈ 0.036ms
实际: 加上循环开销 ≈ 5-10ms ?

vs Triton 20.5ms → 可能 2-4× 加速
```

**前提**:
- ✅ 编译器修复动态循环 bug
- ✅ 修复多块污染 bug
- ✅ 实现生产级 wrapper
- ✅ 多核调度优化

### 中期方向

1. **块并行方案**: 
   - 预计算所有 chunk 的转移矩阵
   - 并行前缀扫描
   - 需要批量 matmul 支持（被同一 bug 阻塞）

2. **预填充/解码分离**:
   - decode (T=1) 单 kernel 无分块
   - 必要的生产优化

3. **混合方案**:
   - Triton K1 + AscendC K2？
   - 利用各自优势

---

## 🎯 实用建议

### 立即使用
```python
from kda_bt16 import kda_bt16_fwd

# ✅ 生产就绪
o, ht = kda_bt16_fwd(
    q, k, v,
    use_fused_kernel=False,  # 推荐两段式
    lower_bound=-5.0          # 已验证
)
```

### 何时考虑 AscendC
```
等待条件:
  1. 编译器 bug 修复 ✅
  2. 端到端 benchmark > 2× Triton ✅
  3. 生产级 wrapper 实现 ✅
  4. 完整测试覆盖 ✅

预计时间: 未知（依赖上游修复）
```

---

## 📚 参考文档

- 完整审查: `CODE_REVIEW_SUMMARY.md`
- AscendC 状态: `ASCENDC_STATUS.md`
- 实验日志: `aclab/README.md`
- Roadmap: `README.md` (行 457-512)

---

## 总结

| 维度 | Triton 两段式 | AscendC M1 | 推荐 |
|------|--------------|-----------|------|
| **端到端性能** | 20.5ms | ~35.5ms | ✅ Triton |
| **Kernel 效率** | 19us/步 | 0.07us/步 | AscendC |
| **生产就绪** | ✅ | ❌ | ✅ Triton |
| **可维护性** | ✅ | ⚠️ | ✅ Triton |
| **可移植性** | ✅ | ❌ | ✅ Triton |
| **未来潜力** | 1.4× | 2-4×? | 🤷 |

**当前选择**: Triton (稳定、可用、性能足够)  
**未来评估**: AscendC M2 (需编译器修复 + 验证)

