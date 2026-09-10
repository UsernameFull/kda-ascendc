# FLA KDA 后端实现深度分析

> 分析对象：`/workspace/kda/fla`（flash-linear-attention main @ `6d80721`，2026-08-17 快照）
> 目标：逐 kernel 拆解 `chunk_kda`（CUDA Triton / Triton-Ascend / FlashKDA / TileLang）的调用图与数据流，输出可执行的 Ascend C 实现清单。

---

## 0. 结论摘要

1. **#1110 dispatch bug 在当前 main 已修复**：`fla/ops/backends/__init__.py` 的 `dispatch` 装饰器现在内部自执行 `torch.compiler.disable(wrapper)`，装饰器顺序问题不再存在。装好 `flash_kda` 包后，满足 verifier 的推理调用会真实走到 FlashKDA。
2. **KDA forward 不是单 kernel**：默认 Triton 路径由 **9+ 个 kernel 串成**（含 3 个在 `chunk.py` 层的前处理 kernel），中间 tensor 全部落 HBM；这是性能主要浪费点。
3. **Ascend 后端已完整（fwd+bwd），但 kernel 拆分比 CUDA 更碎**：fwd intra 拆成 3 个 kernel（CUDA 是 2 个），bwd intra 拆成 3 个（#1130），bwd WY 拆成 6 个（#1074）。这些拆分是 UB 容量约束的产物，也正是 Ascend C 最大的优化空间。
4. **对 Ascend C 的建议**：参考 #915 的两阶段 fusion（BT=16）+ FlashKDA 的 state 片上驻留，不要逐行翻译现有 Triton-Ascend。

---

## 1. 环境与快照

| 项 | 值 |
|---|---|
| 仓库 | https://github.com/fla-org/flash-linear-attention |
| 本地路径 | `/workspace/kda/fla` |
| HEAD | `6d80721` [Fix] Honor the real layouts of y and dy in compute_dh0_kernel (#1139) |
| KDA 相关近期历史 | #1047（Ascend KDA backend）、#1049/#1050（Ascend chunk_fwd_o）、#1055（Ascend chunk_delta_h fwd）、#1065/#1074（Ascend WY bwd）、#1090（Ascend fused_recurrent fwd）、#1099（Ascend recompute_w_u_fwd）、#1113（Ascend tl.dot 左操作数防护）、#1130（Ascend bwd intra 拆分 3 kernel + 1D core-grid） |

### KDA 后端注册表（`fla/ops/kda/backends/__init__.py`）

| Backend | priority | 可用条件 | 覆盖的 dispatch 函数 |
|---|---|---|---|
| `TritonAscendKDABackend` | 0 | `IS_NPU` | fwd/bwd intra、token_parallel、WY fwd/bwd、gate 全部、recurrent fwd |
| `FlashKDABackend` | 3 | `flash_kda` 包 + `FLA_FLASH_KDA!=0` | 仅顶层 `chunk_kda` |
| `KDATileLangBackend` | 5（默认） | `tilelang` + nvcc + `FLA_TILELANG!=0` | 仅 `chunk_kda_bwd_wy_dqkg_fused` |

> 注意：dispatch 是**按函数粒度**的。`chunk_kda_fwd_intra`、`kda_gate_chunk_cumsum` 等各自带 `@dispatch('kda')`，所以 NPU 上即使顶层 `chunk_kda` 不命中任何特殊 backend，内部子函数也会分别路由到 triton_ascend 实现。

---

## 2. 代码地图

```
fla/ops/kda/
├── chunk.py                    # ChunkKDAFunction + chunk_kda 公共 API（dispatch 入口）
├── chunk_fwd.py                # forward 编排：gate→intra→state→output
├── chunk_bwd.py                # backward 编排：dAv→dhu→dqkg→intra→gate
├── chunk_intra.py              # CUDA Triton：inter_solve_fused / bwd_intra / sub_chunk
├── chunk_intra_token_parallel.py  # 非 safe_gate 对角块（token 并行）
├── wy_fast.py                  # WY 表示：w/u/qg/kg 的 fwd 生成 + bwd 准备
├── gate.py                     # KDA gate（softplus 或 lower_bound sigmoid）+ chunk cumsum
├── fused_recurrent.py          # decode 路径单 token recurrent kernel
├── naive.py                    # torch 参考实现
└── backends/
    ├── flash_kda.py            # FlashKDA CUTLASS 包装 + 严格 verifier
    ├── tilelang/               # 仅 chunk_kda_bwd_wy_dqkg_fused 的 TileLang 版
    └── triton_ascend/          # 全套 NPU kernel（gate/intra/wy/bwd/recurrent）
```

跨模块复用的 kernel（重要，Ascend C 移植时也要覆盖）：
- `fla/ops/common/chunk_delta_h.py` → `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`：state propagation（h, v_new, final_state）
- `fla/ops/gla/chunk.py` → `chunk_gla_fwd_kernel_o`：output 计算
- `fla/ops/common/l2norm`、`fla/ops/common/gate`：`l2norm_fwd` / `fused_beta_sigmoid`（chunk.py 层的前处理）

---

## 3. Forward 调用图（默认 CUDA Triton，BT=64, BC=16, D=128）

### 3.1 kernel 序列

```
chunk_kda(q, k, v, g, beta)  [chunk.py:176]
│
├─ (use_qk_l2norm_in_kernel) l2norm_fwd(q) ─┐ 独立 kernel
├─ (use_qk_l2norm_in_kernel) l2norm_fwd(k) ─┤ 独立 kernel
├─ (use_beta_sigmoid_in_kernel) fused_beta_sigmoid(beta)  独立 kernel
│
└─ ChunkKDAFunction.forward
   └─ chunk_kda_fwd  [chunk_fwd.py:20]
      ├─ [gate.py] kda_gate_chunk_cumsum
      │     kernel: kda_gate_chunk_cumsum_vector_kernel
      │     grid:   (cdiv(S,BS), NT, B·HV)
      │     产出:   g_cumsum [B,T,HV,K] fp32（log2 域 chunk 内 cumsum）
      │
      ├─ [chunk_intra.py] chunk_kda_fwd_intra（@dispatch → NPU 有独立版）
      │   ├─ (safe_gate) chunk_kda_fwd_kernel_intra_sub_chunk
      │   │     grid: (NT, NC, B·HV)      # NC = BT/16 = 4
      │   │     产出: Aqk 对角 [B,T,HV,BT], Akkd 对角 [B,T,HV,16] fp32
      │   │
      │   ├─ (非 safe) chunk_kda_fwd_intra_token_parallel（同产出）
      │   │
      │   ├─ chunk_kda_fwd_kernel_inter_solve_fused
      │   │     grid: (NT, B·HV)
      │   │     计算: 非对角 Aqk/Akk（衰减修正的 QK^T、KK^T）
      │   │          + 非对角合并 + 16×16 三角求逆合并 → Akk 全逆
      │   │     产出: Aqk 非对角 [B,T,HV,BT], Akk [B,T,HV,BT]
      │   │
      │   └─ [wy_fast.py] recompute_w_u_fwd_kda_kernel
      │         grid: (NT, B·HV)
      │         产出: w = A·(k·exp2(g)·β)     [B,T,HV,K]
      │               u = A·(v·β)              [B,T,HV,V]
      │               qg = q·exp2(g)           [B,T,HV,K]
      │               kg = k·exp2(g_last-g)    [B,T,HV,K]
      │
      ├─ [chunk_delta_h.py] chunk_gated_delta_rule_fwd_h
      │     kernel: chunk_gated_delta_rule_fwd_kernel_h_blockdim64
      │     grid:   (cdiv(V,BV)·N·HV,)     # 1D，按 V 块×batch×head
      │     recurrent over chunks:  h_t = g·h_{t-1} + w·h_{t-1}·… (delta rule)
      │     产出: h [B,NT,HV,K,V], v_new [B,T,HV,V], final_state
      │
      └─ [gla/chunk.py] chunk_gla_fwd_o_gk
            kernel: chunk_gla_fwd_kernel_o
            grid:   (cdiv(V,BV), NT, B·HV)
            产出:   o = q·h + Aqk·v_new      [B,T,HV,V]
```

### 3.2 中间 tensor 生命周期（全部 HBM 往返）

| tensor | shape（B,T,H=HV=96,K=V=128, BT=64） | dtype | 写 | 读 |
|---|---|---|---|---|
| q_norm / k_norm | [B,T,H,K] | bf16 | l2norm | intra |
| beta_sig | [B,T,HV] | bf16 | beta sigmoid | intra |
| g_cumsum | [B,T,HV,K] | fp32 | gate+cumsum | intra、w/u、fwd_h、o |
| Aqk | [B,T,HV,BT] | bf16 | sub_chunk / inter | o、bwd |
| Akkd | [B,T,HV,16] | fp32 | sub_chunk | inter（fp32 保精度） |
| Akk | [B,T,HV,BT] | bf16 | inter | w/u、bwd |
| w / u | [B,T,HV,K] / [B,T,HV,V] | bf16 | recompute_w_u | fwd_h |
| qg / kg | [B,T,HV,K] | bf16 | recompute_w_u | fwd_h（仅 training） |
| v_new | [B,T,HV,V] | bf16 | fwd_h | o |
| h | [B,NT,HV,K,V] | bf16 | fwd_h | o、bwd |
| o | [B,T,HV,V] | bf16 | o kernel | 输出 |

**D=128、BT=64 时每个 chunk 的片上量级**：Q/K 各 16KB、g 32KB（fp32）、V 16KB、Akk 8KB、h 32KB（fp32 state）——全部常住 UB 需要约 120KB+，这就是为什么 CUDA 版选择拆 kernel、而 Ascend C 的 K1/K2 两阶段设计必须做 buffer 生命周期复用。

---

## 4. Backward 调用图（training）

```
chunk_kda_bwd  [chunk_bwd.py:435]
│  (disable_recompute=False 时先重算)
├─ kda_gate_chunk_cumsum          # g_org → g（重算）
├─ recompute_w_u_fwd              # w,u,qg,kg（重算）
├─ chunk_gated_delta_rule_fwd_h   # h, v_new（重算）
│
├─ [chunk_bwd.py] chunk_kda_bwd_dAv
│     kernel: chunk_kda_bwd_kernel_dAv        grid: (NT, B·HV)
│     dAqk = do·v_new^T (masked), dv = Aqk·do
│
├─ [chunk_delta_h.py] chunk_gated_delta_rule_bwd_dhu
│     kernel: chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64
│     产出: dh [B,NT,HV,K,V], dh0, dv
│
├─ [chunk_bwd.py] chunk_kda_bwd_wy_dqkg_fused
│     kernel: chunk_kda_bwd_kernel_wy_dqkg_fused   grid: (NT, B·HV)
│     产出: dq, dk, dv2, db, dg, dAkk
│     （TileLang backend 只替换这一个 kernel）
│
├─ [chunk_intra.py] chunk_kda_bwd_intra
│     kernel: chunk_kda_bwd_kernel_intra
│     grid:   (NK·NC, NT, B·HV)   # NK=cdiv(K,32), NC=BT/16
│     产出: dq, dk, db, dg（intra 部分）
│
├─ chunk_local_cumsum(reverse)    # dg 反 cumsum
└─ kda_gate_bwd                   # dg, dA, dbias（gate 反传）
```

bwd 是**全局序列相关的**（h 的 backward 有跨 chunk recurrent 依赖），这是 #1128 TileLang 端到端只有 1.05–1.10× 的原因：单独加速 1–2 个 kernel 吃不动整个图。

---

## 5. Triton-Ascend 与 CUDA Triton 的差异（NPU 特有约束）

### 5.1 前向 intra：CUDA 2 kernel → NPU 3 kernel

NPU 版 `chunk_kda_fwd_intra_npu`（triton_ascend/chunk_intra.py:537）把对角求逆**单独拆出**：

```
CUDA:  sub_chunk ────────→ inter_solve_fused（内含对角 solve）
NPU:   sub_chunk → diag_solve（新 kernel）→ inter_solve_fused（只做合并）
```

原因：CUDA 的 fused 版在 NPU 上 UB 放不下（inter kernel 需要同时持有 q/k/g/β 多个 BC×BK fp32 tile）。

### 5.2 反向 intra：#1130 拆成 3 个 kernel + 1D core-grid

```
chunk_kda_bwd_kernel_intra_dq_db_npu        # past 部分 + 对角 → dq, db
chunk_kda_bwd_kernel_intra_dkt_future_npu   # future sub-chunk 贡献 → dkt
chunk_kda_bwd_kernel_intra_dk_dg_npu        # dk, dg 汇总
```

- 全部用 1D `(num_core,)` grid + `task_num/num_core` 循环（AICore task loop），而非 3D grid。
- 注释给出的 UB 账目（chunk_intra.py:672-678）：BC 从 16→32（NC 4→2），BK 分别取 128/128/256，peak live set 控制在 192KB UB 内。
- 编译 BK=256 能过但运行时 UB OOB → 回退 BK=128，说明 **NPU 上 UB 是硬边界，不能照搬 CUDA 的 tiling**。

### 5.3 反向 WY：#1074 拆成 6 个 kernel

```
v_part → k_part → dw_part → dA_mask → dA_mid → dA_finalize
```

CUDA 是单个 `chunk_kda_bwd_kernel_wy_dqkg_fused`。

### 5.4 统一约束

| 约束 | 来源 |
|---|---|
| `chunk_size ∈ {32, 64}`（CUDA 也是） | verifier |
| grid 每轴上限（`ASCEND_MAX_GRID_DIM`） | ascend_ub_manager |
| 单次 launch block 预算 4096（AICore task time） | `_KDA_LAUNCH_BLOCK_BUDGET`，host 分片多 launch |
| `num_warps` 固定 2，无 autotune | NPU kernel |
| UB 容量按 mem_mult 估算 BK/BV（sub:6×, inter:14×） | `compute_row_tile_block_size` |
| 部分中间保持 fp32（Akkd 对角、g_cumsum） | 数值 |

---

## 6. FlashKDA / TileLang backend 的 verifier 限制

### FlashKDA（backends/flash_kda.py:41，推理专用 CUTLASS）

| 条件 | 值 |
|---|---|
| 推理模式 | `torch.inference_mode()`，grad 关闭 |
| dtype | bf16 必须 |
| 维度 | K=V=128 必须 |
| GVA | 不支持（HV==H） |
| kernel 内融合 | `use_qk_l2norm_in_kernel` + `use_gate_in_kernel` + `use_beta_sigmoid_in_kernel` 全 True |
| state 布局 | `state_v_first=True` |
| 其他 | `safe_gate=True`，不支持 CP / return_intermediate_states |

满足时一个 CUTLASS kernel 完成 norm+beta+gate+intra+state+output 全流程（K1/K2 两 kernel 的 CUTLASS 实现）。

### TileLang（backends/tilelang/__init__.py）

- 仅覆盖 `chunk_kda_bwd_wy_dqkg_fused` 一个子算子；
- 不支持 GVA（`v.shape[2] != k.shape[2]` 直接拒绝）；
- 完整 training 加速仍在 PR #1128（open，未合入）。

---

## 7. 性能数据汇总（来自上游 PR/benchmark，非本机实测）

| 对比 | 环境 | 结果 | 来源 |
|---|---|---|---|
| FlashKDA vs Triton chunk_kda（inference fwd） | H20 BF16 D=128 | **1.85–2.31×** | MoonshotAI/FlashKDA BENCHMARK_H20.md |
| #915 fused Triton vs baseline | H800 | 平均 **~1.6×**，小 shape 最高 4.25× | PR #915 |
| #915 TLE vs Triton fused | H800 T8192 H96 D128 | ~1.49–1.52× | PR #915 |
| #915 TLE vs FlashKDA | 同上 | ~1.08–1.19× | PR #915 |
| #915 TLE vs chunk_kda | 同上 | ~2.6–2.8× | PR #915 |
| TileLang 四子算子（intra fwd/dAv/dqkg/intra bwd） | B200 | 1.13–1.54× | PR #1128 |
| TileLang 端到端 fwd+bwd | B200 | **1.05–1.10×** | PR #1128 |
| #1054 小 fusion | B200 D64 | ~1.30×；D128 仅 ~1.05×（→ 关闭） | PR #1054 |
| FlashKDA training（fwd+bwd） | RTX 5090 | 1.03–2.17× shape 依赖（fwd 单独 0.93–0.98×） | PR #1112（open） |

要点：
- **inference 的收益来自 pipeline-level fusion（#915/TLE/FlashKDA 路线），不是 kernel 逐个换实现**；
- D64 小 fusion 收益大而 D128 消失 → 当前 FLA 的中间 kernel 拆分在 D128 下 launch/HBM 占比已经很高；
- Ascend 目前**没有公开可对比的 latency 数据**。

---

## 8. 可执行的 Ascend C 实现清单

### 8.1 目标收敛（第一版）

```
BF16, K=V=128, BT=16（或 64/4 子块）, safe_gate=True,
Hq=Hv（MHA）, inference/prefill only, T≥2K, state_v_first
```

### 8.2 目标架构：两 kernel（对应 FlashKDA K1/K2 与 #915）

```
K1 = norm + beta + gate + cumsum + QK^T + Akk 求逆 + w/u 生成   （token/chunk 并行）
K2 = state propagation (h_t) + v_new + output o                 （head 并行，chunk recurrent）
```

### 8.3 K1 具体设计

- 对照 kernel：CUDA `intra_sub_chunk` + `inter_solve_fused` + `recompute_w_u` 三者的计算，NPU 上目前是 5 个 kernel（sub_chunk/diag_solve/inter_solve/wy_fwd + 外部 l2norm/beta/gate cumsum）。
- 目标：**1 个 Ascend C kernel**，内部 AIV/AIC 分工：
  - AIV：L2Norm、beta sigmoid、gate + exp、cumsum（vector）
  - AIC：QK^T、KK^T（Cube，16×16 基础对齐）
  - AIV：mask、decay 修正、16×16 solve（Akkd）、W/U 生成
- BT=16 的理由（对照 FlashKDA 官方 deep-dive + Ascend Cube 16×16 基础单元）：
  - `128 = 8×16`，QK 全在 Cube 单次 tile 内；
  - 16×16 求逆代价极低，且不需要 sub-chunk 二次分解；
  - 当前 NPU 用 BT=32/64 + BC=16，UB 被迫拆 3 kernel——BT=16 直接消除 diag_solve 的独立性。
- 中间 tensor 消除：q_norm/k_norm/g_cumsum/Aqk/Akkd 全部只留 UB 内（按 3.2 生命周期表复用 buffer），仅 w/u（+kg/qg 视需求）写 GM。

### 8.4 K2 具体设计

- 对照 kernel：`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` + `chunk_gla_fwd_kernel_o`，NPU 上各自独立（#1049 已做 Ascend chunk_fwd_o 融合）。
- 目标：**1 个 Ascend C kernel**，state 全程驻留片上：
  ```
  load h0 (或零) 到 UB  [K,V]=[128,128] fp32 = 64KB（910B UB 192KB 可容纳）
  for chunk in 0..NT-1:
      v_new = v - kᵀ·h_prev      （AIC：k^T·h，需要 state_v_first 布局 [V,K]）
      h_new = g·h_prev + w·h_prev·…   （delta rule，AIC）
      o     = qᵀ·h_new + Aqkᵀ·v_new   （AIC + AIV）
      store o
  store final_state
  ```
- state_v_first=True 时 `h[V,K]` 与 `v_new = v - kᵀ·h` 天然对齐 Cube 连续访问，**必须用 v-first 布局**（这也正是 FlashKDA verifier 强制 state_v_first 的原因）。
- 实验项：state 存储用 bf16 + fp32 累加，把 64KB 降到 32KB，换取双 buffer 或更大 BT。

### 8.5 实施顺序与验证门

1. **Baseline**：`python -m benchmarks.ops.run --op chunk_kda`（registry.py:338 默认参数已是 safe_gate=True, lower_bound=-5）在 NPU 上跑，记 total + 各阶段（可加 `disable_recompute=True` + profiler 分 kernel）。
2. **Triton-Ascend 两阶段原型**（BT=16）：先证明 #915 式 fusion 在 NPU 数值正确（对照 naive.py / test_kda.py），量化收益。
3. **Ascend C K1**：正确性 → GM→UB（DataCopy 对齐）→ Cube util → buffer 复用 → double buffer。
4. **Ascend C K2**：重点 state layout 与驻留；对比 v_first/v_second 两种布局。
5. **联调**：K1→K2 只保留 w/u（+kg/qg）两个中间 tensor；全链路对照 `chunk_kda` 输出（max_abs/max_rel < 1e-2 量级对照 bf16 baseline）。
6. 扩展：varlen → D64/D256 → training/backward → GVA。

### 8.6 优先级排序（按预期收益）

| 优先级 | 动作 | 依据 |
|---|---|---|
| P0 | K2 单 kernel + state 驻留（消除 h/v_new 的 GM 往返） | FlashKDA 1.9–2.3× 主要来源 |
| P0 | K1 单 kernel + BT16（消除 diag_solve 独立 kernel + gate/norm 外部 kernel） | #915 1.6× |
| P1 | AIC/AIV/MTE 三级流水（chunk i 的 Cube 与 chunk i+1 的 Vector 重叠） | 对话方案 4 |
| P1 | 消除 qg/kg（推理路径 disable_recompute 下不生成） | 代码：仅 training 需要 |
| P2 | bwd 三阶段（intra/WY/dhu）各自 fusion | #1128 显示端到端收益有限 |
| P3 | D64/D256、varlen、GVA | 后续 |

### 8.7 不要做的事

- 不要逐行翻译现有 Triton-Ascend kernel → 它已被 UB 约束拆碎，翻译即继承拆分。
- 不要照抄 FlashKDA CUTLASS 的共享内存/寄存器技巧 → NVIDIA 专属，Ascend 上没有对应物。
- 不要用 BT=32/64 起步 → 与 Cube 16×16 不对齐，UB 压力大。
- 不要只 benchmark 单 kernel → 端到端才是验收标准（#1128/#1054 教训）。

---

## 附：关键文件索引

| 文件 | 行号 | 内容 |
|---|---|---|
| fla/ops/kda/chunk.py | 176 | chunk_kda 入口（@dispatch） |
| fla/ops/kda/chunk_fwd.py | 20 | forward 编排 |
| fla/ops/kda/chunk_intra.py | 43 / 683 / 395 | inter_solve_fused / sub_chunk / bwd_intra |
| fla/ops/kda/chunk_intra_token_parallel.py | 50 | 非 safe 对角 |
| fla/ops/kda/wy_fast.py | 34 / 144 | w/u fwd / prepare_wy_repr_bwd |
| fla/ops/kda/gate.py | 375 | gate+chunk cumsum |
| fla/ops/kda/chunk_bwd.py | 43 / 134 / 435 | dAv / wy_dqkg_fused / bwd 编排 |
| fla/ops/kda/fused_recurrent.py | 34 | decode kernel |
| fla/ops/kda/backends/flash_kda.py | 41 | FlashKDA verifier |
| fla/ops/kda/backends/triton_ascend/chunk_intra.py | 138/191/280/778/903/977 | NPU fwd/bwd intra 6 kernel |
| fla/ops/common/chunk_delta_h.py | 58 / 687 | fwd_h / bwd_dhu |
| fla/ops/gla/chunk.py | 345 / 962 | output kernel / chunk_gla_fwd_o_gk |
| fla/ops/backends/__init__.py | 161 | dispatch（已修 #1110） |