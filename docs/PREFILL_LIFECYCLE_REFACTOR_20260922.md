# KDA Prefill 全流水生命周期重构计划

日期：2026-09-22　状态：设计阶段（Level 0/1 已产出可复算的账本，生产 kernel 未改）

目标：在公式、dtype、同一 head 内的递推顺序和现有精度门禁不变的前提下，重构
`pre_gram -> solve -> K2` 的数据生命周期与流水调度，减少中间张量落 GM、扩大有效阶段重叠，
并且**不再用增加 UB buffer 的方式**解决性能问题。

本文件是这一轮的设计书；实测量与判决追加在
`docs/ASCENDC_V1_REFACTOR_PLAN_20260913.md` §11.28 及之后。

## 1. 前提（已实测，不再重复论证）

```text
pre_gram       4.22 ms        （isolated, tools/probe_stage_overlap.py）
solve:AIV      2.40 ms
solve:AIC      2.59 ms
K2             4.06 ms
隔离总和      13.27 ms
```

真实依赖下，三段之间**还没被吃掉的**理想重叠上界约 1.1 ms（§11.27：PG‖K2 0.70 +
PG‖solve:AIV 0.22 + PG‖solve:AIC 0.17），而 `solve:AIV‖K2 = 1.71` 那块属于 solve 自己
已经吃掉的预算，不算新增空间。`K2‖K2 = 1.86` 说明 K2 是延迟受限，但生产形态已是一波
24 块铺满，没有第二份可以并上去。

**结论：本轮的收益必须来自"少搬一次/早搬一次"，不能来自"再排一层流水"。** 任何候选的
预期收益都要先与 1.1 ms 这个上界对照，并且要扣除 §11.12 的实测口径：1 GB 中间流量在
C=64 时只值 0.2 ms（全在 L2），不是 HBM 带宽价。

当前约束（候选必须逐条满足，缺一条即回到设计）：

- K2 `C=64` 的 UB 名义占用约 176.5 KB / 192 KB，实测安全余量约 4 KB；
- A2/A3 上 Cube/Vector 跨 stage 不假设存在可用的 L0C→UB 直通，跨 pipe 数据默认通过稳定
  workspace 交接；
- 不删除必要的 `CrossCoreFlag`、`PipeBarrier` 或 producer/consumer 依赖；同步优化必须通过
  数据流审计与数值 gate；
- 第一阶段不改变公式、不改变 state dtype、不改变单 head 内 chunk 顺序。

## 2. 重构原则

### 2.1 以生命周期划分 stage

不按"一个 matmul 一个 stage"机械拆分。每个中间量必须有以下记录（字段与生成器一致，
见 `docs/artifacts/stage_lifecycle.csv`）：

| 字段 | 要求 |
|---|---|
| name | 缓冲区名（与 kernel/api 里的标识符对应） |
| stage | producer 所在 stage：host / pre_gram / solve / k2 |
| where | GM workspace、L1 resident、L1 scratch、UB、L0 |
| shape/dtype/bytes | 由几何与 dtype 推出，不手写 |
| producer | 哪个 kernel、哪个 pipe 产生 |
| consumer | 哪个 stage、哪个 pipe 消费 |
| scope | `(batch, chunk, head)` 或 head-group |
| lifetime | 单 tile、单 window、单 launch |
| sync | event、CrossCoreFlag、PipeBarrier 或无同步 |
| layout | row-major、packed、fractal 及 dtype |
| role | input / handoff / resident / scratch / output / debug |

账本由 `tools/gen_stage_lifecycle.py` 生成，并且**每一行都要通过引用校验**：行内引用的符号
必须出现在它引用的源文件里、交接点的代码片段必须逐字出现在 `api.py` 或对应 kernel 里；
字节列与 workspace 总量同 `tools/gen_ub_l1_budget.py` 对账。手工维护的表格是"错误 flag id
被验证通过"的来源，这一条不放开。

### 2.2 区分 resident、scratch 和 ping-pong

不能把延长 scratch 生命周期当成 resident。布局必须显式分区：

```text
L1 resident: 跨 stage/window 重复消费的 operand
L1 scratch : 当前 Cube tile 的临时输入
GM ring    : 跨 pipe 的稳定 workspace slot
UB zones   : 本地 vector 输入、计算、输出的分时复用区
```

K2 的 UB 分时复用（`uA..uE/udec` 在 stage2|stage4 两个相位别名）是现有代码里唯一已经把
peak 压下来的机制；新候选只能沿用这个思路，不能新增常驻对象。

### 2.3 用稳定 slot 交接数据

跨 stage 的数据先完整写入稳定 slot，再发布 ready；consumer 只在 stage/window 入口等待，
不在 row/tile 热循环里反复握手。

双 window 使用 ring：`slot = (window_index & 1) * n_local + local_stage_slot`。

`n_local` 是该 stage 对实际要交接的缓冲区个数（当前账本：`pre_gram -> solve` 2 个、
`solve -> k2` 1 个、`pre_gram -> k2` 3 个）。**深度不等于 4 的对（例如 3 个 local slot 就是
6 槽）必须重新设计 credit/free 协议**：每个槽要有自己的 producer-frees / consumer-uses 计数，
不能继续套用简单 ping-pong。

### 2.4 先释放资源，再增加并行度

当前 UB 紧张。所有新增双缓冲必须以**释放旧 buffer** 为前提，不能直接增加常驻对象。
每次候选必须提交：

```text
UB peak <= 192 KB
L1 resident + scratch <= 可用 L1
event/flag 数量合法
```

三条准入数由 `tools/gen_ub_l1_budget.py`（UB/L1/L0/workspace 账）与
`tools/gen_stage_lifecycle.py`（生命周期、slot、GM 流量账）生成，随改动一起更新。

## 3. 分级推进

```text
Level 0  保留现有三段 kernel，完成生命周期和 slot 建模
Level 1  workspace 池化、固定布局、减少重复 transpose/copy
Level 2  pre_gram -> solve 按 window 流式交接
Level 3  solve -> K2 按 window 流式交接，并重构 K2 的本地 lane
Level 4  仅在 Level 2/3 有稳定收益后，评估 device-side persistent scheduler
```

### Level 0（已完成，见 §11.28）

- `docs/artifacts/stage_lifecycle.csv`：44 行、三段全量中间量的 8 字段账本；
- `docs/artifacts/stage_slot_map.csv`：跨 stage 交接与双 window ring 的 local slot 分配；
- `docs/artifacts/stage_traffic.txt`：每次调用的 GM 读写字节约 9.68 GB（C=64，
  `[1,8192,96,128]`），按 producer/consumer 与 stage 拆分；
- `docs/artifacts/k2_sync_audit.{txt,json}`、`ub_l1_workspace_budget_c{16,32,64}.txt`；
- `tests/test_stage_lifecycle.py`：账本的引用校验、字节对账、debug-only 写集合、slot 唯一性。

### Level 1

- **workspace 池化：已冻结**（§11.28 第 2 节）。host 侧 slack 约 5 ms，池化能省的 0.30 ms
  不可能出现在端到端数上；而且池化原型在"同形共享"与"短 buffer"两种错法上都会给出错误结果。
- **固定布局 / 减少重复 copy：第一项已落地**。账本抓出三处 debug-only 写（`Aqk32` 的 AIV
  回写、`A32`、`BetaOut`，合计 405.8 MB/调用）；两个 kernel 加 `debugStores` 尾参数、生产传 0，
  同进程交错 A/B 实测 **-0.164 ms**（10.765 → 10.601 ms，输出与 final_state 逐位一致，
  §11.29）。生产 GM 写从 5042.65 降到 4636.85 MB/调用。
- 同一账本上剩下的候选是 `Aqk32` raw 的那一次复读（201 MB，需要把 mask/scale 搬到 Cube），
  估计 0.04～0.10 ms，要先动 pre_gram 的 AIV band 循环。
- 布局侧：`VnewT`（转置 store）、`Aqk16`/`Aqk32`（[c,CHUNK,CHUNK]）、`Rk/Rv/Qg/Kg/W/U`
  （[c,CHUNK,D] → Nd2Nz）是唯一还在跨 stage 的四种布局；Level 1 只允许**减少转换次数**，
  不允许引入新布局（例如 K2 的 `d1/d2/d3` 与 `vnew_t` 的 task-major 顺序不要动，
  §11.24 的 C=32 bug 就是跨带 walk 与布局假设不一致造成的）。

## 4. 判决实验（2026-09-23）

设计阶段之后，三条路线里已经跑完两条的判决，实测量与完整口径在
`docs/ASCENDC_V1_REFACTOR_PLAN_20260913.md` §11.30 / §11.31。

### 4.1 state/output 分离（路线 1）：不采用

`kernels/v1/k2_state_loop.cpp`（串行状态链，按 chunk 发布入口状态 `H[c]`）+
`kernels/v1/k2_out_parallel.cpp`（全并行 `O[c] = Q[c]H[c] + A[c]Z[c]`），
`k2_mode="split_state_out"`，只用 experimental 入口；K1 一字未动。

| 口径（C=64，`[1,8192,96,128]`，同进程交错 A/B，3 轮 `do_bench` 中位） | fused | split |
|---|---:|---:|
| 端到端 | 10.599 ms | **12.301 ms（+1.70 ms，+16.1%）** |
| K2 内部（profile，含 sync） | 4.078 ms | 3.446 + 2.373 ms |
| K2 的 GM 字节（几何） | 5140.1 MB | 5542.8 MB（+402.7 MB，全是快照读侧） |

两臂**逐位一致**（out 与 final_state），所以这是纯成本判决：状态链只减重 15%，而并行输出核
单独要 2.373 ms（38.6 GFLOP + ~1.85 GB GM 往返 ⇒ ~16 TFLOPS 有效）。**输出留在状态链里**；
唯一的翻盘形态是 Level 4 的设备侧重叠，不是换 tile。

### 4.2 稠密仿射递推（路线 2）：数值可行但形态受限，代价不回收

`tools/probe_affine_precision.py` 用同一组 fp32 中间量对比四种 rounding discipline
（C=64、T=8192、H=8，long/mid/init 三种 regime）。结论：

- 把 `diag(d)` 折进 bf16 的 `E`：状态误差比现实现高 30～70% ⇒ **不合格**；
- `decay` 留在 AIV 的 fp32 状态、只把 `E_off = -Kg^T W` 舍成 bf16：状态误差比现实现**低**
  25～30%（少交 d1 与 Z 两次舍入）⇒ 合格，但跨核交接一点没少；
- 代价侧：+59 GFLOP 按 §11.30 实测价位值 1.5～3.7 ms，而 K2 全部只有 4.08 ms。

### 4.3 分层 C128（本轮新增第一条）：装不下，而且上一步的 K1 已经在倒亏

两部分都在 `docs/ASCENDC_V1_REFACTOR_PLAN_20260913.md` §11.33 有完整口径，这里只留判决：

- **装不下（实测）**：`KDA_CHUNK=128` 全链路 RTC 编译通过，但第一个 launch
  `kda_pre_gram_mix` 直接打挂设备（507015 aicore exception）。per-launch bisect 确认是
  该 kernel 自己。`tools/gen_ub_l1_budget.py` 新增的 `pre_gram_ub()`（从 kernel 源码的
  `InitBuffer` 表达式求值）给出原因：AIV 半边 274.6 / 192 KB（**+82.6**）、L0A 128/64、
  L0B 80/64、L0C 128/128；C=64 的 185.3 / 192 KB 与 kernel 注释里的 "under 8 KB
  headroom" 对上。要装下必须把 `post_gram` 的 band staging 切半、把整 chunk 的 gate
  cumsum 改成带 carry 的 band（加法顺序变，需重跑数值 gate）、并把 L0 操作数切片减半。
- **不回本（实测）**：'chunk 数减半、M 翻倍'这个实验在上一步已经量过：C=32 → C=64 的
  K1 **+0.51 ms**（§11.28 同版本交错 A/B；本轮同机同日 K1-only 复测 +0.316 ms），K2
  −0.94 ms。K1 的增长机制是算术的（Gram/post_gram 的元素数与 solve 的递归 lane 数都随 M
  上升），所以 C64 → C128 不会让 K1 变好。**按判决规则：K1 没有收益 ⇒ 不动 K2，冻结 C128。**

### 4.4 融合式宽 RHS solve（本轮新增第二条）：AIV 地板已经超过整段预算

`tools/probe_rhs_substitution.py` + `kernels/v1/k1_solve_rhs_probe.cpp`（计时探针），同进程
12288 个 chunk-instance、MIN of 5，只写 32 B 防 DCE：

| 臂 | 指令/chunk | ms |
|---|---:|---:|
| shipped recursion floor | 124 | 1.169 |
| fused 2×32（256 lane RHS） | 496 | **5.854** |
| fused 4×16（256 lane RHS） | 240 | **2.941** |

同进程重放生产 solve：AIV 2.123 / AIC 2.587 / 重叠 **2.644 ms**。两个融合形态的 AIV 地板
（不含 RHS gather、不含 W/U store、2×32 还不含耦合 Cube 步）已经是整段的 1.11× / 2.21×。
⇒ **停止，不写 Cube 半边。** 机制：递归的指令数由分块定，指令宽度由另一个操作数定；inverse
是 32 lane × 8 实例，RHS 是 256 lane × 4 实例（同样 128 KB UB），所以每 chunk 指令数 ×4。
副产品：现实现 AIV 的 2.123 ms 里递归只占 1.169 ms，剩下 ~0.95 ms 是 gather/cast/store——
以后要动 solve 应该先动这 0.95 ms。

### 4.5 checkpoint / replay（本轮新增第三条）：被实测的 GM 价位判死

用 §4.1 的 split 实验实测出的价格（+402.7 MB = +1.70 ms ⇒ **4.2 ns/byte**）：interval 4 省
回 ~1.27 ms 的快照流量，但要再读 W/U/Qg/Kg/Aqk（~908 MB ⇒ +3.8 ms）并重算 Z、A@Z、Q@H
（~39 GFLOP ⇒ +2.4 ms），净 **+5 ms 量级**，且 final state 仍只能串行。⇒ **不做 interval
sweep**；重开条件只有一个——诊断显示快照 workspace/流量本身是瓶颈，而本轮测量正好相反
（把快照流量开到最大的 split 变体已经更慢）。

### 4.6 还没有实测的

路线 5（segment scan）仍是专用分支（小 B/H、超长 T），入场条件见 §11.32；C128 与融合
solve 的"组合"随各自路线一起冻结。真要再动 solve，第一步是先削 §4.4 里那 0.95 ms 的
非递归开销，而不是重排递归。

### Level 2 / Level 3 的准入条件

在动手前必须先提交：

1. 该 stage 对的 slot 字节账（每槽字节 × ring 深度 ≤ 可用 L1+UB 余量）；
2. credit/free 协议（谁 set、谁 wait、prologue 是否 priming、最大 backlog）；
3. 同步审计 diff（flag/barrier 数量与配对，含"删除的每一个都要说明为什么安全"）；
4. 精度 gate：与现有 `run_chunk_matrix.sh`（C=16/32/64）位一致或通过 `--gate`；
5. 同进程交错 A/B：先用 §11.28 的 1.1 ms 上界对照，再用实测替换估计。
