# AscendC v1 重构方案（`[1,8192,96,128]`）

日期：2026-09-13　基线：`3115ead`（e2e 14.82 ms）　设备：Ascend 910B3 / CANN 9.1

对照：FLA `chunk_kda` 在 FLA 的 H100/H200 CI 上是 2.722 ms；FLA 的 triton-ascend
后端在本机同一 shape 上是 54.4 ms（chunk32）/ 69.6 ms（chunk64）。我们 14.82 ms
对 FLA-NPU 领先 3.7x，对 FLA-H100 落后 5.4x。本文只规划本仓库 AscendC 路径。

## 0. 摘要

| 阶段 | 目标 | 改动性质 | 前提 |
|---|---:|---|---|
| 现状 | **14.82 ms** | — | 437eaff + 3115ead |
| T0 | 修 3 个已知故障 | bugfix | — |
| T1 | **10–11 ms** | 结构不变（3 段），把 AIV 的工作搬走、把 launch 合掉 | T0 |
| T2 | **6–8 ms** | 结构改变：K2 单波化 + 两侧指令瘦身 | T1 + P2/P3 探针 |
| T3 | **3–4 ms** | 算法级：Cube 常驻 state / 分段两趟扫描 | T2 + 新代数 |

四条硬边界（实测）：

| 边界 | 数值 | 依据 |
|---|---:|---|
| HBM 拷贝带宽 / elementwise | 1165 / 1070–1113 GB/s | `/tmp/roof.py` |
| Cube bf16 | 301 TFLOPS @8192³（265 @4096³） | 同上 |
| 现数据流搬运量 ~5.8 GB | **5.0 ms** | 逐 kernel 流量累加 |
| 算法 FLOPs ~93 GFLOP | **0.31 ms** | 现在实际只有 ~6 TFLOPS（2%） |
| 理想数据流（q/k/v/g/beta/o 各一次 ~1.0 GB） | **0.95 ms** | 下界参考 |

结论：14.82 ms 既不是算力受限也不是带宽受限，而是 **AIV 指令发射 + 跨核协同时延受限**。
因此所有收益都来自"减少 AIV 指令数"或"减少每 chunk 的跨核往返"，
换算法（更少 FLOPs）基本没有收益空间；换数据流（少搬内存）只有 ~1 倍空间。

## 1. 现状分解

`KDA_PROFILE=1`（MIN of 5，`KDA_PRE_GRAM=aiv`）：

| stage | kernel | launch | 网格 | ms |
|---|---|---|---:|---:|
| K1 融合 | `kda_pre_gram_kernel` | 1 | 6144 blocks × 8 chunk | 6.53 |
| K1 求解 | `kda_solve_wu_wide` + `kda_solve_wu_cube_kernel` | 2 | 1536 + 12288 | 1.90 |
| K2 递推 | `kda_k2_persistent_loop` | 1 | 48 blocks = 24 AIC × **2 波** | 6.42 |

后续每条杠杆都要引用这些实测数字：

- **K1**：每 chunk 固定 133 ns 发射成本（与 H 无关，`[1,1024,4]` 0.46 ms → `[1,8192,96]` 6.53 ms）；
  MTE 管道几乎空闲（`aiv_mte2_ratio` 0.066）而 vec 0.845；每 chunk 61 KB 流量 → 3.0 GB，
  实际 ~300 GB/s（带宽的 26%）。Gram 循环占单 block 2.21 ms 里的 1.05 ms（47%）。
- **K2**：每 chunk-step 6.3 µs（nh=2 = 2 个 head），合每 head-chunk **3.15 µs**；
  同一结构在 `[1,8192,32]` 上 nh=1 是 5.9 µs/step，说明 heads/block 的摊销是真实的
  （1 波 512 步的地板 ~2.7–2.8 ms）。data-free 协议探针：4-phase 7.2 µs/step、
  2-phase 3.6、1-phase 1.85；但把某个 phase *整个删掉* 只值 ~1%（阶段被另一侧掩盖）。
- **solve**：`wide` 已到带宽地板（370 GB/s，40 MB/pass）；Cube 部分 NC≤4 是硬约束
  （L0A/L0B 槽位索引 `pass*NC + ch`，L0C/`qc` 队列只有 NC 深）。

## 2. 杠杆清单（按证据强度排序）

| # | 杠杆 | 预期 | 证据 |
|---|---|---|---|
| L1 | K1 的 Gram 上配对 Cube（mix） | 6.53 → ~4.0 | 已实现（437eaff）：Gram 47% of block、publish 只值 0.047 ms、AIC 参考吞吐 0.78 µs/chunk |
| L2 | solve 两次 launch 合成一个 MIX，a16 不再落 GM | 1.90 → 1.2–1.4 | `[1,8192,32]` 隔离测数：wide 0.10 ms、Cube 0.55 ms（NC≤4 上限）；H=96 时 a16 写+读 50 MB |
| L3 | K1 AIV 指令瘦身（prep 复用 Qg/Kg、少一次 exp2、合并 reduce） | 6.53 → 5.6–6.0 | 450 指令/chunk 的发射成本模型；`pre_gram` 的每指令 ~27 cycle |
| L4 | K2 AIV 瘦身（state 更新用 `MulAddDst`、S16 发布合并、gather 合并） | -0.5–0.8 ms | 每 head-chunk 3.15 µs 里 state 的 64+64 repeat 是最大单项 |
| L5 | K2 单波化（MAXH=4 / nblk=24） | 6.42 → 3.5–3.8 | 96 头 / 24 AIC 必须 2 波；nh=2 的 AIV TBuf 154.5 KB / 192 KB，需 UB diet |
| L6 | 分段两趟扫描（T 方向并行 + 段仿射传播） | 见 §5 | 需要新代数；Cube 只多 0.7 ms 的活 |
| L7 | Cube 常驻 state（decay 折进 Q/K operand） | 见 §5 | 未验证，需要 P4 探针 |

## 3. T0：先修三个已知故障（0.5–1 天）

1. **`KDA_PRE_GRAM=mix`（437eaff，当前默认）在 `[1,8192,96,128]` 与 `[1,8192,32,128]`
   不返回**（507014 超时两次后 >500 s 挂死）。修不好之前默认值退回 `aiv`（一行），
   mix 改显式 opt-in。
   - 先做 shape 二分：`[1,256,96]`、`[1,1024,96]`、`[1,2048,96]`、`[1,8192,64]`、
     `[1,8192,96]`，把断点定到 "H 相关" 还是 "grid 大小相关"。
   - 可疑点：AIV 双 subcore 的 2-deep ring 在长网格下的 WAR；`FL_DONE` 的
     1 set / 2 wait 广播语义；AIC 侧 `PipeBarrier<PIPE_ALL>` 的位置（对照
     `k2_persistent_loop` 的结论：缺它 1/3 概率 aicore exception）；
     `pre_unroll=8` 时一个 block 16 chunk 的流水深度。
2. **`persistent_loop` 的 507015**（本 shape 约 1/3 概率）——不是性能问题，但污染所有
   benchmark 与 CI 结果。
3. **Triton `kda_bt16_fwd` 在 H=96, T≥2048 的 K1 挂死**（grid ≥12288）：修，或在
   `kda_bt16_fwd` 的 docstring / README 里明确不支持该 shape。

T0 完成判据：`KDA_PRE_GRAM=aiv` 路径 20 次连续 e2e 无 fault；`pytest -q` 全绿；
`benchmarks/bench_fla_compare.py --shape 1,8192,96,128` 复现 14.8 ± 0.3 ms。

## 4. T1：结构不变，10–11 ms（1–2 周）

三条并行、互不阻塞的改动，每条独立提交 + 独立测数：

### T1.1 落地 mix（L1）
- 修 T0.1 的挂死；在小 shape 上先跑 `pytest` + 与 `aiv` 路径逐输出比对
  （`return_intermediates` 的 13 个输出，期望除 Aqk/L 的 rounding 外 bit-identical）。
- 目标：`pre_gram_ms` 6.53 → 4.0–4.5；`aiv_vec_ratio` 0.845 → ~0.45，
  AIC 保持 <30% duty。
- 失败模式记录：一旦发现某个 shape 挂死，先查 ring 深度和 `FL_DONE` 广播，
  再查 AIC 的 `PipeBarrier`。

### T1.2 solve 融合（L2）
- 新的 MIX kernel：AIV 做 `wide` 递归（32 chunk/lane），AIC 做 `W`/`U` 的 Mmad，
  两者在同一 block 里流水；`a16` 只走 GM 0.5 KB/chunk（或 L1），`a32` 只在
  debug 时写。
- 约束：NC ≤ 4（L0A/L0B 槽位索引）；每个 pass 必须重新发自己的 B load（`Rk`/`Rv`）。
- 目标：`solve_ms` 1.90 → 1.2–1.4。

### T1.3 K1 指令瘦身（L3）
- 复用 `Qg`/`Kg`（`preprocess` 已物化 gated 值）而不是重算 `2^gc`/`2^-gc`；
  把两次 row-reduce 合并；检查 `PipeBarrier` 的实际必要性（向量管道 in-order，
  之前测过删掉无收益，但 **发射槽位** 才是成本）。
- 目标：6.53 → 5.6–6.0（与 T1.1 叠加后 3.5–4.0）。

T1 完成判据：e2e ≤ 11 ms，`out_err < 1e-3`、`state_err < 1e-4`（`tests/test_persistent_loop.py`
的门禁口径），并在 `[1,8192,32]` 上确认无回归（历史数字 13.67 ms 全 pass）。

## 5. T2：结构改变，6–8 ms（3–6 周）

T2 的两个必要条件是 **K1+solve ≤ 3.5 ms** 且 **K2 ≤ 3.3 ms**，对应两条主攻线：

### T2.1 K2 单波化（L5）
96 头 / 24 AIC，`MAXH=2` 时 48 block = 2 波，K2 就是 2 × 3.2 ms。若 `MAXH=4`
（nblk=24，1 波），同样 512 步 → **3.5–3.8 ms**（每 step 从 6.3 µs 涨到 ~7 µs，估；
这条必须先有 P2 的 step 分解）。
拦路的是 UB：nh=2 时 AIV TBuf 已 154.5 KB / 192 KB（`us` 64 KB + staging 90 KB）。
需要 UB diet，按代价从低到高：
1. `us16`（32 KB）不常驻，改成每步复用 `d2f/d3f` 的临时区；
2. staging 缓冲（`uu/uv/ut/uo/ud1` 各 2 KB，`uof/ud1f/...` 各 4 KB）合并复用；
3. 实在不够就把 fp32 state 分两半处理（半个 64 KB 在 UB，另一半在 L1/UB 临时区）。
前置探针 P3（`/tmp/kdaval/ub_probe.cpp` 的扩展版）先验证 4 份 state + staging 能声明。

### T2.2 两侧指令瘦身（L4 + L3 的延长线）
- K2：state 更新 `state = state*decay + d4` 现在是 64 `Mul` + 64 `Add`（或等价），
  换 `MulAddDst`（64 条）并把 S16 的 cast/store 合并到同一次遍历；每 head-chunk
  省 ~0.3–0.5 µs → 0.6–1.0 ms。
- K1：把 per-row 的 `Brcb` + 多次 broadcast 合并；确认哪些 `PipeBarrier` 可以
  换成事件对。
- 目标：K1+solve ~3.2–3.5，K2 ~3.0–3.3。

### T2.3 已否决的融合路线（不要再试）
- **不要在 loop 的 AIV 里融合 `pre_gram`**：AIV 的发射是共享、可加的临界资源，
  每指令 ~13 ns/(head,chunk)，`pre_gram` 的 ~450 指令/chunk 会加 ~5–6 µs 到
  6.3 µs 的 step 上（≈ +3 ms > 它作为独立 launch 的 2.21 ms），UB 也不够
  （154.5 + 125.1 KB > 192 KB）。
- **不要用两 stream 分段 overlap**：合法版本（s32 串接）只值 ~1%。
- **不要合并 phase / 合并 per-head ops**：data-free 探针看好，实机更慢
  （per-head coalescing 3.03 → 3.74–3.83 ms）且要退回 `PIPE_ALL` 才不 fault。

## 6. T3：算法级，3–4 ms（research，先探针后代数）

真正把 3 ms 变可能的是把 **AIV 每 chunk 的指令数** 压到接近 0，两条候选：

### T3.1 Cube 常驻 state（L7）
`S ← decay⊙S + d4` 是 128×128 fp32 的整块读改写（每 head-chunk 8192 元素），
是 K2 里最大单项。若把 decay 折进 operand（KDA 的累积 T 矩阵已经在 K1 里算过
同类量），state 就只剩 Mmad 累加，可以留在 L0C/L1 不出片。需要先做的探针：
- P4：`Fixpipe` 能否直接写 L1 / 下一 chunk 能否把 L1 上的 fp32 当 A/B operand
  （非转置、128 宽需要 `Nd2Nz`）；若不行，则"state 只走 L0C→UB→L1"是否仍比现在便宜。
- P5：把 state 放 bf16（每 head 32 KB）是否精度可接受（`state_err < 1e-4` 要重新验）。

### T3.2 分段两趟扫描（L6）
把 T 切成 `seg` 段，第一趟每段以 `S_in = 0` 算出段内输出与段的仿射传播算子
（段内 chunk 变换的复合，128×128 矩阵），第二趟把 `S_in` 传下去并修正输出
（`out += q @ (M_seg S_in)`）。Cube 代价：每 head 约 2×(512/seg) 次 128×128×128
Mmad、共 ~200 GFLOP → 0.7 ms @301 TFLOPS，可接受；换来的是把"每 AIC 2048 个串行
head-chunk step"压成 2048/seg + 传播。
前提：每 step 的成本必须已经降到通讯主导（否则 step 数不变、总时间不变）。

T3 完成判据：`[1,8192,96,128]` e2e ≤ 4 ms，精度门禁不变，且 `[1,2048,96]`、
`[1,8192,32]` 不回归。

## 7. 验证与门禁（每个 milestone 都要过）

1. `pytest -q`（`test_persistent_loop.py` 的 `out_err < 1e-3` / `state_err < 1e-4`）。
2. 与上一版逐输出比对：纯搬运/重构要求 **bit-identical**，数值路径变化要求
   `out_err` 不劣化且记录绝对值。
3. `KDA_PROFILE=1` 的 stage 分解（MIN of 5，同一进程）。
4. `benchmarks/bench_fla_compare.py --shape 1,8192,96,128 --json` 记录 e2e 与 FLA 对照。
5. 连续 20 次 e2e 无 507014/507015 fault。
6. 每个改动把数字写进对应 doc（仓库惯例：一行 commit body + 一节 doc）。

## 8. 已知坑 / 风险

- CANN 9.1 的跨核 loop：flag id ≤ 7、协议迭代不变、prologue 用 pre-set flag；
  loop-carried `SetFlag/WaitFlag` 会挂（`k1_pre_gram_mix.cpp` 与
  `docs/VLLM_ASCEND_KDA_REVIEW_20260911.md` 都有记录）。
- AIC 侧每个 stage 后必须 `PipeBarrier<PIPE_ALL>`（`FIX_M` 不覆盖 L0A/L0B/L0C 复用）。
- `k1_pre_gram_mix.cpp` 对写法敏感：重排 buffer 声明或移动某个 `PipeBarrier<PIPE_V>`
  会触发 aivec error（mte 0x8030860ef）。
- `MIX` 内核里 AIV/AIC 的 UB 预算必须实测声明（`ub_probe`），不要靠推算。
- 本 shape 上 `persistent_loop` 有 1/3 概率 fault：任何性能结论都要先看 fault 计数。

## 9. 探针清单（先测后改，1–2 天量级）

| # | 探针 | 回答什么 | 现状 |
|---|---|---|---|
| P1 | mix kernel 的 `msprof` | AIC/AIV 各自的 duty 与 stall | 有（mix 已带 msprof 结论） |
| P2 | K2 每 step 成本分解（在各 stage 追加 N 条 dummy 指令） | 6.3 µs 里 AIV/AIC/握手各占多少 | 技术已有（`k2slack3/4.py`），未做该 shape |
| P3 | UB 预算（4 份 state + staging） | MAXH=4 是否可声明 | `ub_probe.cpp` 可改 |
| P4 | `Fixpipe` L0C→L1 / state 不出片 | T3.1 是否可行 | 无 |
| P5 | bf16 state 的精度 | 是否能用半精度 state 换 UB | 无 |
| P6 | shape 二分：mix 挂死点 | T0.1 的定位 | 无 |

## 10. K2 实测地板（2026-09-13 晚，14 个探针）

同一 shape、同一进程、MIN of 2–3，基线 `k2_ms = 6.45–6.47`（`MAXH=4`，24 block
= 1 波 = 512 chunk-step = 每 step 12.65 µs = 每 head-chunk 3.16 µs）：

| 探针 | k2_ms | 结论 |
|---|---:|---|
| 基线 | 6.47 | — |
| AIC 两个 stage 的 `LoadData`/`Mmad`/`Fixpipe` 全删 | 6.46 | AIC 算力 = 0 |
| AIV stage2 计算/搬运 + stage4 epilogue/state 全删 | 6.42 | AIV 算力 = 0 |
| AIC 两处 `PipeBarrier<PIPE_ALL>` → `M_MTE1` 事件对（**已落地**） | 6.38 | 屏障 = 0.07 |
| AIV `PIPE_ALL` 删除 / `MTE2_V` 对删除 / `V_MTE2` 对删除 | 6.45–6.49 | 核内事件 = 0 |
| D4 写+读各缩到 1/4（−4.7 GB 流量） | 6.35 | 带宽 = 0.1 |
| AIC 的 GM 加载全删 | 6.23 | AIC 加载 = 0.25 |
| **两侧 GM 加载全删（零数据）** | **5.37** | 剩下全是协议 |
| 所有指针钉到同一 GM 地址 | 10.90 | 同址冲突反向变慢 |
| flag 粗化 16/step → 4/step | 12.14 | **变慢** |
| 粗化 + 零数据 | 9.31 | 粗化丢失 head 间重叠 |
| `nh=1`（96 block，4 波） | 12.39 | 4 × 3.1 |
| `nh=2`（48 block，2 波） | 6.65 | 基本相同 |
| AIC 加载批量化（单 depth-1 大 buffer） | 7.87 | 变慢 |

模型（14 行全对上）：`k2_ms ≈ 512 step × 4 phase × nh × ~0.65 µs`。
每个 head 的 4 个 phase 是 `state(AIV)→AIC→d12→AIV→v_new→AIC→d34→AIV` 四跳；
四跳之所以串行，是因为 stage2/stage4 的 UB staging 被 nh 个 head 共用，代码
结构上必须"所有 head 走完 stage2 才进 stage4"——这也解释了为什么 per-head
flag 有效、粗化反而慢（粗化丢失 head 间重叠，且换不出 phase 数）。

**判决**：
- **L4 / T2.2 作废**：K2 与两侧算力、加载、屏障、事件全部无关（差额 ≤0.25 ms）。
- **L5 / T2.1 已经吃满**（`MAXH=4` → 24 block 单波，收益已计入）。
- K2 地板 = **5.37 ms**（零数据零算力），实际 6.3–6.5 ms。要 3 ms 级只有两条路：
  1. **减少串行 step 数** → T3.2 分段两趟扫描（其前提"每 step 已是通讯主导"现已实测成立）。
  2. **让 head 之间 phase 重叠** → 需要 stage2/stage4 各自独立的 UB（`ud1/uf/ux`
     各拆两份 ≈ +14 KB，可行）**且** AIC 的 L0B 拆两份（state 32 KB×2 + d34
     operand 8 KB×2 = 80 KB > 64 KB，**不可行**）。除非 state 走 bf16（T3.1/P5），
     否则 AIC 侧先卡死。
- 另外：`kda_solve_wu_cube_kernel` 1.61 ms / 12288 block = **512 波**，每 unit
  393 ns（~20 条指令 + 3 次 drain），而它的真实算力只有 0.02 ms。这是 K2 之外
  最值得动的单点。已落地的部分：给每个 (pass, chunk) 一个独立 L0C 槽
  （`cfall`，2*NC*8 KB = 64 KB）后，`Fixpipe → Mmad` 的 `FIX_M` drain 不再需要，
  1.946 → 1.772 ms 且 **bit-identical**（d_out/d_state 都是 0.00e+00）。
  剩下的两个 drain（`MTE1_M`、`M_FIX`）删掉会 fault（实测 507015），要拿它们
  必须真做软件流水（`NC` 加深会被 AIC 的 `TQue<B1, N≥4>` 上限挡住，
  必须改成手工 ping-pong L1），预估还能拿 0.4–0.8 ms。

## 11. R1 落地：C=64 两级 solve（2026-09-13 深夜）

R1 的靶子是 solve 的 `wide`（AIV）侧：C=64 时它的行递推是 `M^3/2`=131k lane/chunk，
实测 3.865 ms 里 3.12 ms 是这一项（`/tmp/wideattr.py`：整段删掉只剩 0.747 ms）。
把每个 chunk 的 64×64 三角拆成 SB=2 个 32×32 对角块，递推的 lane 数按 `SB^2` 掉，
剩下一个耦合块 `X21 = -X22 L21 X11`（两个 32³ 的乘）交给 Cube——
这正是 Cube 该干的活。改动落在 `k1_solve_wu_wide.cpp`（两级的 AIV 侧）+
新 kernel `k1_solve_assemble.cpp`（AIC 侧耦合块）+ `python/kda_ascendc_v1/api.py`（接线）。

### 11.1 先把三个真 bug 摘出来

上一轮的"assemble 会 fault"不是 harness 问题，是两个真错：

1. **尺寸写反**：kernel 里 `constexpr int32_t M = KDA_CHUNK`、`PC = M * SB`，
   但后面到处把 `M` 当*子块*用。`PC=128` 时 L 的 gather 步长是 4 倍，
   读到映射窗口外 → `aivec error … "The GM address accessed by scalar exceeds
   48 bits"`（0x4000）。它的指纹很好认：fault 的核数正好是 grid 的 1/25
   （grid 256 → 50 个 fault，grid 2458 → 123 个），因为只有读写落到窗口外的那
   几个 block 会炸。修正就是把 `M`/`PC` 的角色换回来（`M = PC / SB`）。
2. **`#if SB > 1` 是死代码**：`SB` 是 `constexpr` 不是宏，预处理把它当 0，
   整条 L21/Xb/Lneg 通路被编译掉（所以 `xb`/`lneg` 一直是哨兵值、assemble 没东西可读）。
   6 处都改成 `#if KDA_SOLVE_WIDE_SUBB > 1`。
3. **`Lneg` 的 store 步长错**：`DataCopyParams(NCH, M/16, (RW21-M)/16, …)` →
   `(NCH, M / 16, 0, (MM - M) / 16)`（`l21b` 的 chunk 是连续的，行间没有间隙）。

三个都修完后，NC=4/8/12 在真 shape 上干净且数值精确（见 11.3）。

### 11.2 配置扫描（MIN of 4，一个进程一个配置）

| 配置 | wide | asm | cube | solve |
|---|---:|---:|---:|---:|
| `SB=1, NC=4`（R1 前） | 3.865 | - | 1.290 | 5.19 |
| `SB=2, NC=4, KA=4` | 1.957 | 0.777 | 1.259 | 4.00 |
| `SB=2, NC=6, KA=4` | 1.490 | 0.760 | 1.260 | 3.51 |
| `SB=2, NC=8, KA=4` | 1.351 | 0.772 | 1.258 | 3.38 |
| `SB=2, NC=10, KA=4` | 1.202 | 0.758 | 1.261 | 3.22 |
| `SB=2, NC=12, KA=4` | 1.245 | 0.771 | 1.259 | 3.27 |
| `SB=2, NC=8, KA=4`，16 片双流 | - | - | - | **2.51** |

- `NC`（wide 一个 block 的 tile 列数）到 10 为止，12 回落：UB 在 NC=8 时用到
  118 KB / 192 KB，12 时 172 KB。
- `KA=4` 稳定优于 2（0.76 vs 0.87）——但 assemble 的 L1 队列必须是 `NC` 深，
  2 深的队列在第三次 `AllocTensor` 上**死锁**（不是变慢），`KA=8` 会挂在
  L0 槽位上。
- Cube solve 一直钉在 1.258–1.261（L0C 只能容 2 个 chunk×(pass, chunk) 槽），
  它是这条链的下一个瓶颈。

### 11.3 收益与代价

- **solve：5.19 → 2.51 ms**（隔离测）/ 2.62 ms（流水线里，2026-09-14 复测
  `pre_gram 4.14 + solve 2.62 + k2 1.67 = 8.42 ms`，MIN of 4）。
- **e2e 的 8.42 ms 现在还不能当收益算**：见 11.5，K2 整个家族是 CHUNK=16 实现，
  C=64 下它每个 chunk 只走 16 行（**做 1/4 的活**）并且输出是错的，所以这个 e2e
  数字既偏快又不正确。solve 段的收益不受影响（K1 是 CHUNK 参数化的，且已逐段
  对过 fp64 参照）。
- 数值：fp64 参照下 `X11/X22` 误差 4.882e-04（一次 bf16 舍入），耦合块 1.411e-03。
  numpy 重放同一条 bf16 链（`Xb`、`Lneg`、`P`、`X21` 各自 bf16）两个数一位不差，
  说明误差全部来自 bf16 存储本身，kernel 没有额外贡献；单级路径是"整块一次舍入"，
  这是 4x 更短的递推的代价。
- **双流重叠**：wide 是 `AIV_ONLY`、assemble+Cube 是 `AIC_ONLY`，把 chunk 区间切成
  `lcm(wide block, asm block, cube block)` 的整数倍后交错在两条 stream 上（每片一个
  event），两个引擎同时干活：3.22 → 2.51 ms，输出与串行**逐字节相同**（12288 个
  chunk 的 `a16`/`W`/`U` 全比过）。切片必须按整个 unit 切，否则奇数片会把最后
  一个 chunk 交给会去读*下一片*（还没写）的 Cube unit。

### 11.4 还剩什么

- Cube solve 1.26 ms 是波次受限（12288 block / 24 AIC = 512 波）：L0C 的
  `(pass, chunk)` 槽位上限把它钉在 NC=2，要动它只能真做软件流水或把
  assemble 折进去（下一个 R）。
- assemble 0.76 ms 里每个 block 两次全屏障 + 8 次 `SetFlag/WaitFlag` 是主要成本，
  per-chunk 流水（pass0(ch)+pass1(ch) 交错）是明显可做的下一步。
- 重叠之后 AIC 侧（2.0 ms）成了关键路径，而 wide 只有 1.2 ms：下一步要么把
  cube 压到 1 ms 以内，要么把 AIC 的活再分掉一部分。

### 11.5 为什么 C=64 的 e2e 数字现在还不能算数（2026-09-14 发现）

C=64/C=32 的 chunk size 只在 **K1** 里真正落地：`grep -l KDA_CHUNK kernels/v1/k2_*.cpp`
是空的——K2 全家（`k2_persistent_loop`、`k2_d12*`、`k2_vnew`、`k2_d34`、
`k2_outstate*`）都把 `M = 16 / K = 16 / N = 64` 写成字面量，persistent loop 里
每个 chunk 的 GM 偏移还是 `(bh * NT + chunk) * M * D`（`k2_persistent_loop.cpp:128`）。
KDA_CHUNK=64 时它按 16 行的步子走 64 行的 chunk：每个 chunk 只处理 16 行、偏移全错，
于是"快且错"；`k2_mode="separated"` 同错。

证据（2026-09-14，全部 `MIN of 4` / 单进程）：

| 检查 | 结果 |
|---|---|
| e2e vs fp32 参照 `[1,8192,8,128]`，C=16 | out rel 8.6e-03 ✓ |
| 同上，C=64 | out rel 1.0 ✗（\|ref\| 6.4e-2，我们的值整体不相关） |
| 小 shape vs fp64 参照（T=64/640，H=2），C=16 | rel 7.4e-03 ✓ |
| 同上，C=64 | rel 1.0 ✗（两种 K2 mode 都一样错） |
| K1 逐段 vs fp64 参照，C=64 | `Qn/Kn/Qg/Kg/Rk/Rv/Aqk32/Aqk/L/Decay` 全部 bf16 级 ✓ |
| `KDA_SOLVE_WIDE_SUBB=1`（R1 前）C=64，同一输入 | **vector core exception**（既有 C=64 缺陷） |

本轮顺带修掉的两个 K1 真 bug（都是"输出看运气"级别，e2e 快照看不出来）：

1. **两级路径的 Cube 启动没有按真实 chunk 数截断**：`rk/rv/W/U` 只按 `c` 个 chunk
   分配，而 slice 长度 `n` 带了 padding，Cube kernel 又只按传入的 `n` 截断自己的
   循环 → 尾块越过 `W/U` 末尾写 GM（OOB 写），小 shape 直接把邻居 buffer 打花
   （T=64/H=2 输出 1e35）。修法：launch 前 `ncube = min(n, c - lo)`。
2. **`A16`/`A32` 的严格上三角从来没人写**：wide kernel 本来用"一行零 + srcGap=0"
   广播去补，但 `DataCopyParams` 的 gap=0 语义是**连续读**（`/tmp/dcprobe.py`），
   一行源会读到 buffer 外面；于是那块保持 `torch.empty` 的内容——真 shape 的
   100 MB 新分配恰好是 0（所以之前"验证过"），小 shape 回收内存里是 1e36/inf，
   Cube 读到后 W/U 变 inf、输出全错。修法：UB 里放整块 `M x M` 零 tile，
   `A16`/`A32` 都补；另外 `L[c:]` 现在显式清零（kernel 契约本来就要求"尾块解成
   无害单位阵"）。

**结论**：R1（两级 solve）本身已经"做完 + 数值验证 + 变快"，但 C=64 的 e2e
收益要等 **K2 的 CHUNK 参数化**（把 16 行的 tile 改成 `CHUNK/16` 个 16 行子块、
chunk 步长用 `CHUNK * D`，并处理 chunk 内的 aqk 耦合/state 更新）落地之后才能
报。C=16 的 e2e 一直是好的，可作回归基线。

## 11.6. R2 落地：K2 的 chunk 参数化 + C=64 的 e2e 收益（2026-09-14）

R2 的靶子是 11.5 的结论：C=64 的 e2e 数字要等 K2 的 CHUNK 参数化落地才能报。
改动落在 `kernels/v1/k2_persistent_loop.cpp`（全量重写为 `M = KDA_CHUNK` 的公式：
`NG = 2 * BV / M`、stage 3 的收缩维 = chunk 行数、所有 GM 偏移 `(bh*NT+chunk)*M*D`，
没有一条 chunk-size 分支）和 `python/kda_ascendc_v1/api.py`（`PERSIST_MAXH`
随 chunk size 取 4/2，见 11.6.2）。

### 11.6.1 三个 tile 布局事实（都不是"把 M 从 16 改成 64"）

1. **`Vt` 的 store 必须是行主序。** `AscendC::Transpose` 是 16×16 原语，stage 2
   的转置输出是 *packed 块序*（块 `(j0,m0)` 落在 `bl = j0*(M/16)+m0`）。`M=16`
   时 packed 序就是行主序（C=16 一直对就是这个原因），`M=64` 时必须把块散回去
   （`FR` 个"一行 16 元素"的 burst，目的行距 `M`）。AIC 侧按**行主序**读
   （每个 band `Nd2NzParams(1, FR, K, 0, K, FR, 1, 0)`），拿到 packed 序不会
   fault、只会算错：对齐 fp32 参照 out rel 1.0。隔离硬件探针
   `/tmp/vtprobe*.py`（5 种形态逐位一致）、dump 网格 `/tmp/vtgrid3.npz`。
2. **d4 的 A 操作数（L0A）是"每个 value tile `BV/FR` 个 band"，不是 `NB`。**
   该操作数是 `[NG*M, K]`、两个 value tile iv-major，所以 `lv` 每个 `iv` 要
   `BV/FR` 个 band；`NB = M/FR` 只在 C=64（`M = BV`）恰好相等。C=16 时
   `NB=1` 只装了 1/4 的操作数——这个错是 11.6.3 的对照实验抓到的。
3. **`K == FR` 值得单开一条 load 路径。** 每行正好一个 C0 块时 ND 布局*就是*
   fractal 序，一条 plain burst 顶得上 general 路径的逐 band `Nd2Nz`
   （16 个 descriptor/band vs 整块 ~2 个）：C=16 上这一条值 K2 整体
   15.6 → 5.5 ms。

### 11.6.2 收益：C=64 要靠"两个 head/block"才真赚

`MIN of 4`，`[1,8192,96,128]`，一个进程一个配置：

| 配置 | pre_gram | solve | K2 | e2e |
|---|---:|---:|---:|---:|
| C=16，MAXH=4（旧默认） | 4.550 | 2.105 | 5.529 | 12.201 |
| C=32，MAXH=2 | 4.316 | 2.288 | 5.941 | 12.546 |
| C=64，MAXH=1（初次能算对） | 4.159 | 2.591 | 6.042 | 12.792 |
| C=64，MAXH=2（R2 默认） | 4.156 | 2.621 | 3.913 | **10.742** |

C=32 顺带做了对照（同一个内核的 general 分支、`NB=2 != BV/FR=4`，数值同样是
8.620e-03/4.245e-03）：它比 C=16 慢、比 C=64 慢，所以默认还是 C=64。K2 的
每 chunk-head 成本随 chunk 变大而升（C=16/32/64 各 ~2.7/5.8/7.7 us），
说明大 tile 的*效率*在下降——但 chunk 数目的下降仍然更快。

C=64 只减少"每 chunk 的 flag 往返"（512 → 128），而**K2 不是 flag 受限、
是 descriptor/工作受限**：§10 的 protocol 模型（`k2 ~= chunks × 4 phase × nh ×
0.65 us`）预测 K2 该掉 4 倍，实测只从 6.51 掉到 6.04。真正让 C=64 赚的是
**第二个 head 把两个引擎重叠起来**——深度一协议下 `nh=1` 时 AIC 的
`d12(h+1)` 没有下一个 head 可重叠，两个引擎严格串行：K2 6.04 → 3.91 ms，
e2e 12.79 → 10.74 ms，且输出**逐位不变**、stage gate 的 rel 一字不变。
UB 账：staging 112.5 KB（M=64）+ 2×32 KB state = 176.5 KB / 192 KB，第三个
head 要 208.5 KB（UB 溢出是 kernel 侧 aivec error，不是 host 侧分配失败）。

### 11.6.3 控制实验抓到的 C=16 回归（回写教训）

重写只动"切 chunk 的方式"，但 C=16 的对照跑出 out rel 7.59e-02 / state rel
1.04（HEAD 是 8.62e-03 / 4.25e-03）——不是"读到陈旧 build"，是真回归：
`lv` 的 band 数被写成 `NB`（C=64 恰好等于 `BV/FR`，所以 C=64 对、C=16 错）。
修法（`BV/FR`）+ `K == FR` 快路径后，C=16 回到 8.620e-03 / 4.245e-03，并且比
旧 kernel 略快（K2 6.51 → 5.53）。教训：把 tile 公式从"单一 chunk size 下恰好
成立"推广时，**必须同时在旧 chunk size 上跑一遍控制**；C=16 的 separated 路径
正好是免费的 oracle。

顺带一个对照：pre-R2 的 C=64 e2e 是 8.24 ms，但每个 chunk 只走 16 行（做 1/4
的活）——"快且错"；现在 10.74 ms 才是这个几何的真实数字。

### 11.6.4 数值门禁（可复现）

- e2e vs 主机 fp32 参照，`[1,8192,8,128]`，C=16 与 C=64 都是
  out rel **8.620e-03** / state rel **4.245e-03**（两个 chunk size 一位不差）。
- K2 stage gate（numpy 重放同一批 K1 中间结果，C=64 T=64 H=2）：
  `d1` 6.562e-36、`d2` 2.585e-36、`vnew`/`vnewT`/`d3` 0、`out` 9.591e-04、
  `d4`/`state` 3.294e-08。
- pytest `tests/test_persistent_loop.py`：C=16 全绿（6 passed）；C=64
  3 skipped（`k2_mode="separated"` 是 C=16-only oracle）+ 3 passed。新增
  `test_persistent_loop_matches_fp32_reference` 是 chunk-generic 的（C=32/64
  也有门禁，pre-R2 的"快且错"它抓得住），determinism 用例改成 48 heads
  （任何配置都 `nh >= 2`，能压到多 head 交错的那条路）。

### 11.6.5 现在的账与下一步

`10.742 = pre_gram 4.156 + solve 2.621 + K2 3.913`。对照 FLA：本机
triton-ascend 54.4 ms（领先 5.1×），FLA H100/H200 CI 2.722 ms（落后 3.9×）。
四个阶段的排序没变，但 K2 从"最大头"降到第二位（pre_gram 现在是第一大段）：

1. **K2 的 descriptor 数**：AIC 侧每个 chunk-head 约 640 条（W/Qg/S16 的
   逐 band `Nd2Nz` 各占 64/64/128），这是 nh 重叠之后剩下的主项——把"整块一次
   转换"的等价形式找出来（先确认 `dstNzC0Stride` 的整块语义）就能再砍一半。
2. **K2 的 `nh` 上不去**：112.5 KB staging + 2×32 KB state = 176.5/192 KB。
   要 4 heads/block 得先缩 staging（AIV 只 stage 自己那 64 列、d4 与 S16 的
   两半进一步共用），否则 C=64 就钉在 2。
3. **pre_gram 4.16 ms 现在是最大单段**：回到 §4 的 L3（指令瘦身）与 L2（融合）
   那条线。

## 11.7. K2 分解探针：3.9 ms 到底花在哪（2026-09-14 下午）

§11.6.5 把"descriptor 数 / 载入量"列为 K2 的第一杠杆。17 个变体（同一个进程里按不同符号名
编译多份、发射时按名切换，输入与进程状态共用；脚本 `/tmp/k2split{,2,3,4}.py`）把这条线否了。
数值本来就是错的（探针只测时间）。同进程 stock 的抖动是 3.824–3.875（跨进程），所以只有
同进程 A/B 的差值有意义：

| 变体（C=64，`[1,8192,96,128]`，MIN of 3） | k2_ms | Δ |
|---|---:|---:|
| AIC 的 GM→L1 载入全关（W/Qg/S16/Aqk/KgT/Vt） | 3.689 | **−0.15** |
| W/Qg/S16 换**同体积 plain copy**（去掉 Nd2Nz 转换） | 3.874 | +0.04 |
| AIV 的 GM→L1 载入全关（U/D1/D2/D3/D4/Decay） | 3.613 | **−0.22** |
| AIC 的 L0 载入全关（`LoadData`/`LoadDataWithTranspose`） | 3.805 | −0.07 |
| AIV 的向量计算全关（Cast/Sub/Mul/Add/Muls/Transpose/Duplicate） | 3.734 | −0.14 |
| 全部 `Fixpipe` 关 | 3.657 | −0.18 |
| 只关 D4（128×128 fp32）的 `Fixpipe` | 3.683 | −0.15 |
| 全部 `Mmad` 关 | 3.982 | +0.15（噪声：Cube 完全被藏住） |
| **AIV 的四个 store 全关（V/Vt/Out/S16）** | **2.625** | **−1.21** |
| **载入 + store + 向量计算全关（只剩 flag/Mmad/fixpipe/L0）** | **1.460** | **−2.42** |
| 只去掉非转置 `V`（vnew）的 store（同进程 A/B） | 3.879 | +0.06（中性） |

1. **不是载入 / descriptor 受限。** 两个引擎的全部 GM→L1 载入加起来 0.37 ms（AIC 0.15 +
   AIV 0.22），Nd2Nz 转换本身 0（同体积 plain copy 无效），L0 载入 0.07，AIV 向量计算 0.14，
   而 Cube 的 Mmad 去掉反而慢 0.15——它完全藏在别的管道后面。"把 W/Qg/S16 换成整块 Nd2Nz、
   砍 descriptor 数"这条线可以从计划里划掉。
2. **AIV 的 store 段是唯一显著项（1.21 ms，31%），但不是字节数线性。** 单独去掉其中 20%
   （`V` = 16 KB/chunk-head）实测 **0**（§11.7 最后一行，已在 R2 上试过并回退）。所以这 1.21 ms
   是"store 序列 + 它两侧的 WAR `PipeBarrier<PIPE_ALL>` 与 MTE3→V 事件对"的**整段**代价，
   不是搬运量；D4 的 64 KB/chunk-head 的 fixpipe 写入也只值 0.15 ms。
3. **地板 1.46 ms = 每 chunk 的机器成本。** 512 个串行 head-step（2 波 × 128 chunk × 2 head）
   → 2.85 µs/step ≈ 4 phase × 0.7 µs，§10 的 protocol 模型在 C=64 上仍然成立。注意各项单独
   拿掉之和（0.15+0.22+0.07+0.14+0.18 = 0.76）远小于组合拿掉（2.42）：各段互相掩盖，
   K2 的 3.9 ms 是**每 chunk 的串行段数**，不是任何单项的带宽或指令量。

对 R3 的含义：能动的只剩结构，且都不属于"少搬几个 KB"——
- 把 state/d4 的 GM 往返搬进 L0C（T3.1）：同时砍掉 fixpipe 的 64 KB/chunk-head 和 AIV 的 d4 读；
- 减少每 chunk 的 store/屏障段数：stage 2/4 的"两半"合并、去掉一次 MTE3→V 事件对，或把
  `Vt` 的转置搬到 AIC 的 `LoadDataWithTranspose`（省掉 AIV 的 16 次 `Transpose` 与整个 Vt store 段）。

## 11.8. R3（一）：K2 的 UB 分期复用，把 MAXH 从 2 抬到 4（2026-09-14 晚）

§11.7 说 K2 只能靠结构动刀，而结构里最便宜的一刀是**把每块里的 head 数从 2 抬到 4**：
b=1/h=96 时 96 个 head 落在 24 个 AIC 上就是**一波**，而 MAXH=2 时 48 块 = 两波，
第二波要把 128 个 chunk 整个再走一遍。这件事之前被 UB 顶着：fp32 state 32 KB/head 常驻，
4 个就是 128 KB，剩下的 64 KB 装不下当时 112.5 KB 的 staging。

**做法：staging 按阶段复用。** stage 2（v_new）和 stage 4（out + state 递推）在本 loop 里
从不同时活着——每一段开头都有一个 `PipeBarrier<PIPE_ALL>` 在排水——所以一套 buffer 可以
轮着用两段：

| buffer | 字节（C=64） | stage 2 | stage 4 |
|---|---:|---|---|
| A | `TILE * 4` = 16 KB | `vf`（u 的 fp32 展宽） | `of`（out 的 fp32 累加） |
| E | `TILE * 4` = 16 KB | `sc`（缩放后的 k，bf16）+ `vt` | `d1f/d2f/d3f`（fp32 展宽暂存） |
| B | 8 KB | `ub` | `d2`，随后是 s16 的 1/4 |
| C | 8 KB | `d1` | `d3`，随后是 `ob` |
| D | 8 KB | `vb` | `d4` 的 16 行 1/4 |

56.5 KB，加上 128 KB 的 state = 184.5 KB / 192 KB。代价是 stage 4 的递推必须从
"两个 32 行半"改成"**四个 16 行 1/4**"：`uD` 只有 8 KB，装不下 32 行的 fp32 d4
（16 KB）。同进程 A/B（交错 MIN of 4，输出与 fp32 state **逐位相同**）：

| [1,8192,96,128]，CHUNK=64 | k2_ms |
|---|---:|
| HEAD（112.5 KB staging，MAXH 2，48 块 = 2 波） | 6.232 |
| 新 layout，但仍是 MAXH 2（48 块 = 2 波） | 6.774（**+0.54**：四个 1/4 的代价） |
| 新 layout + MAXH 4（24 块 = 1 波） | **5.973**（−0.26） |

CHUNK=32 同向（6.118 → 5.808），CHUNK=16 本来就能上 4。

1. **收益来自"一波 vs 两波"，不是来自少搬字节。** 同分布的 A/B 显示分期复用本身是
   **负的**（+0.54 ms，四个 1/4 比两个 1/2 多出 2 组 `MTE3→V` / `V→MTE2` 往返，
   每 chunk-head 多约 8 次跨 pipe 握手）。所以这个改动不能拆开用：2 个 head 配 56.5 KB
   是白扔（120.5 KB），必须配 MAXH 4 才回本。
2. **K2 的时间是"每 block 的 head-step 数 × 每步延迟"，不是块数。** MAXH 2（48 块）：
   每块 2 head × 128 chunk = 256 步，两波；MAXH 4（24 块）：每块 512 步，一波。
   总 head-step 数不变（12288），所以 1 波 × 512 步 ≈ 2 波 × 256 步，
   实测 5.97 对 2 × 1.97 = 3.94 的差就是第二波的 ramp。
3. **数值门禁**（fp32 参考，H=8，T=8192）：C=16/32/64 全部 `out rel=8.620e-03,
   state rel=4.245e-03`，与改前逐位相同；MAXH 4 与 MAXH 2 的输出与 fp32 state
   在全 shape 上 diff = 0.000e+00。

## 11.9. R3（二）：发射开销不是瓶颈（实测地板 2.8 µs），以及两个小旋钮（2026-09-14 晚）

### 先否掉一个假设：kernel 发射很便宜

前一轮把 solve 的 2.4 ms 归给"48 次发射 + event"。空 kernel 实测（`/tmp/launchfloor.py`、
`/tmp/drain.py`，同一个进程内 200 连发、尾部一次 sync）：

| 场景 | enqueue | 含 drain |
|---|---:|---:|
| 空 kernel，1 块 | 1.69 µs | 2.69 µs |
| 空 kernel，192 块 | 1.80 µs | 2.65 µs |
| 空 kernel，25 个 arg blob | 2.82 µs | 2.96 µs |
| **同一条 stream 上的依赖链**（空 kernel，1/48/192 块） | — | **2.7–3.0 µs** |
| event record + wait_event + wait_stream | 12.9 µs | 18.7 µs |

**依赖链上每次发射 2.8 µs**，和独立连发的吞吐同量级。所以 48 次发射 ≈ 0.13 ms，
不是 2.4 ms；`wait_stream`（~9 µs）才是 event 那侧的真成本。solve 的 2.53 ms 是**真算力**。

### solve 的真结构

按发射序号重放每个 kernel（`/tmp/solvefloor3.py`，C=64，重放 30 次取均值，同一份 args
只改块数；注意超出 kernel 循环长度的块是空转的，所以"时间 ∝ chunk 数"）：

| kernel | 48 块 | 96 | 192 | 384 | 768 | 归属 |
|---|---:|---:|---:|---:|---:|---:|
| `kda_solve_wu_wide`（AIV） | 0.026 | 0.050 | 0.115 | 0.096 | 0.110 ms | **1.84 ms**（16 片 × 0.115） |
| `kda_solve_assemble`（AIC） | 0.010 | 0.019 | 0.035 | 0.040 | 0.054 | 0.56 |
| `kda_solve_wu_cube_kernel`（AIC） | 0.010 | 0.013 | 0.023 | 0.045 | 0.057 | 0.72 |
| 空 kernel 参照 | — | — | — | — | — | 0.003 |

即 AIV 侧 1.84 ms、AIC 侧 1.28 ms，两流重叠后实测 2.53 ms（重叠只吃掉了 0.6 ms，
不是理论上的 1.28）。**wide kernel 是 0.15 µs/chunk**（12288 chunk / 48 AIV 核 =
256 chunk/核 → **7.2 µs per chunk per core**），和 K2 的每 step 7.2 µs 同量级。

**否掉的第二刀：SB=4。** 两级 solve 的 SB 从 2 提到 4 应该把行递归的 lane 从 16 降到 4
（doc 里 64→16→4 的第三档），但实测**更慢**：同进程交错 A/B `solve_ms` 2.769（SB=2）
对 2.876（SB=4），而且 SB=4 变体在进程内直接 NaN（独立进程里数值是好的：out 8.919e-03 /
state 4.245e-03）。所以这条线不是 lane 数的问题。

### 两个抬起来的旋钮

| 旋钮 | 旧 | 新 | 实测（同进程交错 MIN of 3） |
|---|---|---|---|
| `SOLVE_OVERLAP` | 16 | 24 | solve 2.650 → **2.525**（8→2.742，48→3.463） |
| pre_gram 的 `pre_unroll` | `min(8, max(2, c//512))` | `min(64, max(2, c//384))` | pre_gram 4.163 → **4.038** |

`pre_unroll` 的规律是**波数**，不是块数下限：c=12288/C=64 时 768 块（pu 8）= 32 波 →
4.163，384（16）= 4.099，**256（24）= 10.67 波 → 4.162**，192（32）= 8 波 → **4.043**，
96（64）= 4 波 → 4.031，128（48）= 5.33 波 → 4.563，48（128）= 2 波 → 4.160。
每波 ~0.01 ms，且**不满一波按一整波算**（5.33 波和 2.67 波都掉到 4.56）。所以按
`c // 384` 定 unroll 让 C=64 落在 8 波、C=16 落在 384 块（实测 4.551 → 4.078）；
c < 768 时两个公式取值相同，中等形状不动（[1,4096,32,128]/C=64：pu 4 → 0.814，
新值 5 → 0.815，8 → 0.842 ms）。

### 现在的账（[1,8192,96,128]，C=64，MIN of 4，与 §11.8 两项合并后）

`pre_gram 4.051 + solve 2.539 + k2 3.670 = 10.289 ms`（改动前 10.744–10.804）。

新结论：**三个阶段都在 ~7–16 µs/chunk/core 的量级上**——pre_gram 16.3、solve-wide 7.2、
k2 7.2。它们的向量数学都远小于这个数（一个 [64,64] fp32 tile op ≈ 64 cycle），
载入量/descriptor 数也已各自被 §11.7 和 §11.6 否掉。所以剩下的 5 ms 目标不在"搬得更少"，
而在**每 chunk 的标量/协议段数**：谁能把每 chunk 的指令段和握手段砍掉一半，谁就拿走一半。

## 11.10. R3（三）：K2 每 chunk 两次整机排水的 0.65 ms（两次尝试的失败与教训）

§11.7 把 K2 的 1.21 ms 归到"AIV 的 store 段 + 它两侧的 WAR 屏障与事件对"。这一节把
"两侧的屏障"单独拆开测（`/tmp/k2bar.py`、`/tmp/k2drain2.py`、`/tmp/k2min.py`，都是同一个
进程里按符号名切换，MIN of 3）：

| 变体 | k2_ms | Δ | 输出对不对 |
|---|---:|---:|---|
| stock | 3.713 | — | 对 |
| 删掉 stage 4 的那次 `PipeBarrier<PIPE_ALL>` | 3.386 | −0.33 | — |
| 删掉 stage 2 的那次 | 3.180 | **−0.53** | — |
| 两次都删 | 3.065 | **−0.65** | 错（out-diff 1.2e-02） |
| 两次都换成 `PipeBarrier<PIPE_V>` + `PipeBarrier<PIPE_MTE3>` | 3.059 | −0.61 | **错（out-diff 1.18e-02）** |

1. **0.65 ms（K2 的 18%）就在这两次排水上**，而且是每 chunk 每 head 各一次。
2. **换 pipe 屏障等于没换，而且会把结果算错。** `PipeBarrier<PIPE_X>` 只排本 pipe 内
   指令的**发射序**，**不等**异步引擎把这条指令的源数据读完——所以它和"直接删掉"给出
   同一个时间（3.06）和同一个错值。这和 kernel 注释里那条教训（start-up publish 上
   `PipeBarrier<PIPE_MTE3>` 挡不住后面的 V Cast）是同一件事：WAR 只能靠**事件对**。
3. **事件对版本直接挂死**（不是算错，是 hang，需要复位设备）。做法是给 stage 2 加
   `Set/Wait<V_MTE2>`（放在 `Cast(vb, …)` 之后 / 迭代开头）和 `Set/Wait<MTE3_V>`
   （放在 Vt store 之后 / Transpose 之前），再加一对 prologue Set。挂死的疑点是
   **event id 池**：AIV 段已经拿了 4 个（`e2v/ev3/evm2/e3v`，每类一个），AIC 段 5 个，
   再各要一个新 id 可能就超出分配器给这个 block 的池子（拿不到 id 时 Set/Wait 不会报错，
   只会死等）。下一次要做这件事，先把 id 预算算清楚（或严格复用已有 id、保持
   Set→Wait 交替），不要靠 `AllocEventID` 撞运气。
4. **为什么排水是暴露的、而不是被藏住：** 它等的是 stage 4 的 MTE3 store（S16 16 KB +
   Out），而这两个 store 的源恰好就是下一个 chunk 的 stage 2 要写的 buffer
   （`uB`←`ub`、`uC`←`d1`）——§11.8 的分期复用把 stage 2/4 的 buffer 并了，所以这个
   WAR 是"上一 chunk 最后一次 store" 与 "本 chunk 第一次 load" 的正面相撞，
   中间没有别的 AIV 工作可以垫。
5. **两条可能的出路**（都还没做）：
   - 只给 `ub` / `d1` 各留一块专属 buffer（16 KB），就地把排水降级成两个窄事件对；
     现在 184.5 KB / 192 KB 放不下，除非再腾出 16 KB（例如让 `sc` 和 `vt` 共用 8 KB——
     它们都是 8 KB 的 bf16 tile，但 Transpose 的原地写需要验证）。
   - 把 event id 预算做对，换窄事件对（收益同 0.65 ms，但挡不住 store 本身还在排队：
     上限是 0.65 而不是 0.65 的全部）。
   注意两次失败都是"更快但算错"：**这个改动必须先看逐位一致，再看时间**。

第三次尝试（也是唯一安全的一次）：把两次排水各自**挪到 `CrossCoreWaitFlag` 之后**——
排水等的是上一 chunk 的 store，而交叉核 flag 本来就是这个 block 要等的东西，理论上能盖住。
结果**逐位一致但只快 0.04 ms**（3.681 → 3.640，交错的 MIN of 3，噪声 ±0.03），
说明 AIC 早就把 flag 放下了、这个等待里没有可垫的时间。所以 0.65 ms 是**净暴露的串行延迟**，
只能靠"让被等的 buffer 不再是下一轮的源"（专属 buffer / 双缓冲）或窄事件对去掉。

一个附带的否证：pre_gram 的 `PipeBarrier<PIPE_V>` 是**免费**的（删掉全部：4.038 → 4.026，
噪声内），所以"标量屏障太贵"这条猜想不成立——V 管道本来就是顺序的。

## 11.11. R3（四）：Vt 的两种块序——一次 store 换 1.21 ms（2026-09-14 晚）

§11.7 的探针把 K2 的 1.21 ms 钉在"`Vt` 的 store 段（0.82 ms）+ 它两侧的 Nd2Nz 读"上，
并把"连续 store + 连续读"记为**快 1.21 ms 但算错**。这一节把那个"算错"拆开，找出真正
需要的是哪两种块序，然后一次拿回来。

### 两种块序从哪来

`Vt` 存的是 `v_new^T`（[BV, M]），AIC 有两处读它，走的是**不同的分形走法**：

- **d4 的 A 操作数（`lv`）**：`BV / FR` 个 band，每个 band 一次
  `Nd2NzParams(1, FR, K, 0, K, FR, 1, 0)` → L1 里分形序号 = **行块优先**
  `(b, c) → b * 4 + c`（b = 16 行的 band，c = 16 列的块）。
- **d34 的 B 操作数（`lx`）**：一次 `Nd2NzParams(1, BV, K, 0, K, BV, 1, 0)`，
  `dstNzC0Stride = BV` 把源的 C0 块按**列块优先**打包（kernel 头部那句注释）→
  分形序号 = `(b, c) → c * 4 + b`。

两者读的是同一批 16×16 块，只是**块级转置**。所以"一次连续 store"只能满足一个：

| 变体 | `Vt` 的块序 | `lv` 读法 | `lx` 读法 | k2_ms | out-diff | state-diff |
|---|---|---|---:|---:|---:|
| stock | 行主序（16 次跳写） | Nd2Nz band×4 | Nd2Nz 一次 | 3.696 | 0 | 0 |
| A | **A 操作数序**（packed，vt 原样） | 一次 plain burst | 4 次带 stride 的块拷贝 | **2.457** | **0.000e+00** | **0.000e+00** |
| B | B 操作数序（`(m0 * BV/FR + j0)` 散射） | 4 次带 stride | 一次 plain burst | 2.475 | 3.5e+31 | 4.1e+33 |
| C | A 操作数序 | 一次 plain burst | 一次 plain burst（**错**） | 2.472 | 1.132e-02 | 0 |

（同一进程内按符号名切换，4 个变体各编一份、各跑 2 轮取 MIN，`/tmp/k2pack.py`；
`KDA_CHUNK=64`。注意第一次跑忘了设 `KDA_CHUNK`，在 C=16 上四个变体逐位相同——
C=16 时 `K == FR`，store 本来就是连续的、两个读也走 `K == FR` 分支，
这反而**顺带证明了 M = FR 时新旧 store 逐位一致**。）

1. **C 行解释了 §11.7 的"state 逐位、out 错"**：d4 的两个操作数是 `lv` 和 **`lk`（kg^T）**，
   而 `lx` 只喂 d3（out 的 d3 项）——所以块序只错在 B 操作数时，state 完全不受影响。
2. **A 的修法是 4 次 `DataCopyParams(BV / FR, FR * FR / 16, ...)`**：把 (b, c) 块搬到
   `c * BV / FR + b` 的位置——nBurst = 4 个 b、每段 16 块（512 B）、源段距 48 块、
   目的连续。它替掉的是 `Nd2Nz`，块数不变、字节数不变，只是不再"按行重新打包"。
3. **B 变体给出的是纯垃圾**（不是"接近/差一点"）：它的 `lv` 源偏移推错了，
   说明这条路上没有"差一点"的中间态可捡——要么块序对，要么整块错位。

### 落地

`kernels/v1/k2_persistent_loop.cpp`：stage 2 的 store 变成一次 `DataCopy(Vt[out0], vt, BV * M)`；
stage 3 的 `else` 分支里 `lv` 一次 plain burst、`lx` 4 次带 stride 的块拷贝。

`[1,8192,96,128]`、C=64、MIN of 4（`/tmp/final2.py`）：

| | pre_gram | solve | k2 | 合计 |
|---|---:|---:|---:|---:|
| 改动前（§11.9） | 4.051 | 2.539 | 3.670 | 10.289 |
| 改动后 | 4.027 | 2.535 | **2.464** | **9.026** |

数值闸门（`/tmp/refcmp_full.py`，C=64，T=8192，H=8）读数与改动前**完全一致**：
`out rel=8.620e-03 abs=5.490e-04 | state rel=4.245e-03 abs=2.453e-03`；
`tests/test_persistent_loop.py` 6 项（含 C=16/32 的 fp32 参考与确定性）全过。

**K2 的账本现在是 2.46 ms**：0.82 的 store 与 ~0.4 的读已经拿掉，剩下的两个大项还是
§11.10 的两次整机排水（0.65 ms，需要专属 buffer 或窄事件对）和每 chunk 的协议段数。

## 11.12. R3（五）：pre_gram 的账本推翻重写，以及 post_gram 的两级流水（2026-09-14 深夜）

§11.11 之后 pre_gram 是 4.03 ms、占 e2e 的 45%，而之前所有笔记都把它记成
"HBM 受限、3.6 GB/pass"。今天用"同进程内删一块、量一块"（`/tmp/pgdec.py`、
`/tmp/pgcs.py`、`/tmp/pgdrain.py`、`/tmp/pgint2.py`，全部 MIN of 2–3，
C=64、`[1,8192,96,128]`）把这件事测清楚了：**它不是带宽受限**。

### 1. 删掉流量几乎不省时间（推翻"HBM 受限"）

| 变体（只删、不改结构） | 省下的字节 | pre_gram | 差值 |
|---|---:|---:|---:|
| stock | — | 4.041 | — |
| `nopub`：AIV 不写 Ga/Gk/Gb、AIC 不读（96 KB/chunk = 1.18 GB） | 1.18 GB | 3.793 | **−0.25** |
| `nord`：post_gram 不读 Aqk32/L（393 MB） | 393 MB | 4.011 | −0.03 |
| `nowb`：post_gram 不回写 Aqk32/L（393 MB） | 393 MB | 4.014 | −0.03 |
| `dead`：两个 mask 的 load+Compares 全删（400 MB） | 400 MB | 3.942 | **−0.08** |
| `aicdead`：AIC 的 6 组 load + 2 个 Mmad + Fixpipe 全删（80 KB/chunk = 983 MB） | 983 MB | 4.333 | **+0.30（变慢！）** |

读法：**C=64 时 1 GB 的流量只值 0.2 ms 左右**（≈ 5 TB/s 的有效带宽 = 全在 L2 里），
而 Aqk32/L/Aqk16 那圈"Cube→AIV→GM"的往返是 0.03 ms 级、mask 通道是 0.08 ms 级。
`aicdead` 去掉 Cube 的工作反而慢 0.3 ms，说明 Cube 的 48 KB/chunk 读+32 KB/chunk 写
不但完全被掩盖，还替 AIV 的 MTE 队列让出了节奏；**Cube 在这个 kernel 里是免费的算力**。

### 2. 真正的账：9 次整机排水 = 0.80 ms

`PipeBarrier<PIPE_ALL>` 在 pre_gram 里每 chunk 出现 **9 次**（4 次 pass 边界、4 次
post_gram 的 band 边界、1 次 chunk 尾巴）。逐类删掉（timing-only，结果会错）：

| 删掉 | 次数/chunk | pre_gram | 差值 |
|---|---:|---:|---:|
| pass 边界的排水 | 4 | 3.499 | **−0.54** |
| band 边界的排水 | 4 | 3.799 | **−0.24** |
| chunk 尾巴的排水 | 1 | 3.792 | **−0.25** |
| 三者都删 | 9 | 3.244 | **−0.80** |

这些排水的存在理由都是 **WAR**：下一段的 MTE2 load / V write 落进上一段 MTE3 store
还在读的 UB。§11.11 的笔记已经判过"只能用双缓冲 TQue，事件对会挂"，今天按这个方向
落地了 band 那一级（见下）。另外量到：

| 探针 | pre_gram | 说明 |
|---|---:|---|
| `nocs`：删掉 63 条串行 Add 的 gate cumsum | 3.774 | 整条 cumsum = **0.26 ms** |
| `nobar`：cumsum 保留、删掉 63 个 `PipeBarrier<PIPE_V>` | 4.027 | 屏障是免费的（依赖链才是钱） |
| `logcs`：Hillis–Steele 6 步 log-scan 换串行扫描 | 5.064 | **更慢 1.0 ms**，负结果 |

cumsum 只值 0.26 ms，且换成 log-scan 反而更慢（去掉 57 个屏障但把每条 Add 的
地址改成递减遍历），所以这条路到此为止。

### 3. 落地：post_gram 的 band 级双缓冲（TQue）

`kernels/v1/k1_pre_gram_mix.cpp` 的 `post_gram` 从"一套 staging + 尾巴排水"改成
**四条两级队列**：Gram 两块 fp32（`TQue<VECIN,2>`）、两块 fp32 mask、masked fp32 结果
（`TQue<VECOUT,2>`）、bf16 取整结果（`TQue<VECOUT,2>`）。WAR 由队列自己的事件覆盖
（`qout`/`qo16` 的消费者是 MTE3，编译器知道），band 边界的排水删除；
select 的位掩码是 V→V，同一管道内有序，改成一块 512 B 的普通 scratch。

| | pre_gram | solve | k2 | 合计 |
|---|---:|---:|---:|---:|
| 改动前 | 4.027 | 2.535 | 2.464 | 9.026 |
| 改动后 | **3.928** | 2.525 | 2.459 | **8.935** |

逐位一致（同进程 A/B：`out-diff=0.000e+00`、`state-diff=0.000e+00`），数值闸门读数
不变（`out rel=8.620e-03 abs=5.490e-04 | state rel=4.245e-03 abs=2.453e-03`），
`tests/test_persistent_loop.py` 6 项全过。

### 4. 负结果：mask 上提（hoist）与 UB 的真实预算

- mask 的两个三角矩阵是**与 chunk 无关**的，本想把两次 `Compares` 提到 block 开头
  一次算好（省 400 MB 的 mask 读 + 400 万条向量指令）。`inband128` 证明"位掩码放在
  缓冲区偏移处、Compares/Select 读同一偏移"是**逐位正确**的；但把 Compares 搬到
  chunk 循环外面（含专门 buffer、含放到循环第一次迭代里两种写法）**都产出垃圾**
  ——疑似 TPipe 对 TBuf 的活跃期合并（子张量访问不入活跃期分析），负结果记在
  `/tmp/pghoist2.py`、`/tmp/pghoist3.py`。
- **TQue 的 UB 代价远高于名义值**：band 那一级名义 +34 KB，但实测把整核的 UB 余量
  从 ~96 KB 打到 **<8 KB**（活 dummy buffer 二分：stock +96 KB 过、+128 KB 挂；
  加了 band 队列后 +8 KB 就挂）。所以 pass 级双缓冲（名义 +40 KB）**目前在 UB 上放不下**，
  需要先腾地方（把 mask/位掩码改成极小缓冲、或让 scratch 复用）——这是 R3 的下一件事。
- pass 级双缓冲的实现已经在 `/tmp/pgq2.py` 里写好并编过（AI Core Error = UB 用尽，
  与 dummy 探针的失败现象一致），等 UB 腾出来即可复用。

### 5. 当前状态

`[1,8192,96,128]`、CHUNK=64、MIN of 4：**8.935 ms**（pre_gram 3.928 / solve 2.525 / k2 2.459）。
三条线的下一个大项：pre_gram 是 pass 级排水（0.54 ms 上限，卡 UB）；solve 已到重叠地板；
K2 还是 §11.10 的两次排水（0.65 ms，两次替换尝试都挂）。

## 11.13. R3（六）：pass 边界其实是两半，0.49 ms 到手（2026-09-14 深夜）

§11.12 把 pass 边界的 `PIPE_ALL` 记成"MTE3-read → MTE2-write 的 WAR，要用整机排水
才盖得住"。这一轮把它拆开量了：**边界不是一个竞争，是两个**，而且只有一半需要花钱。

### 1. 现象：只留 MTE3→V 排水，坏的是"除最后一遍之外的所有行"

把边界换成自配对的 `SetFlag/WaitFlag<MTE3_V>(e3p)`（V 等 MTE3 落地）后，
`/tmp/pgqF.py`（`b=1,t=64,h=1`、CHUNK=64、NP=4 遍）读数：

| 张量 | 坏元素 |
|---|---:|
| `Qn`、`Kn`、`Qg`、`Kg`、`Rk`、`W`、`d1`、`d2`、`d4`、`A16`、`A32`、`Aqk` … | 0 |
| `Rv` | 6136/8192（**第 0/1/2 遍全坏，第 3 遍全干净**） |
| `U`/`Vnew`/`VnewT`/`d3` | 继承 `Rv` |

"最后一遍干净"是"下一轮迭代把它冲掉"的指纹。三个 MTE2 落地缓冲
（`qnb`/`knb`/`rvb`）都是 V 读、MTE2 写，而 **q/k 在 pass 开头就读、rv 在 pass 中段才读**
——只有晚读者中枪，说明**下一遍的 DataCopyPad 已经跑到这一遍 V 的前面去了**。

### 2. 机理：`WaitFlag` 只停它自己那条管道，不停标量发射

`SetFlag/WaitFlag` 标记的是**管道队列里的一个点**：`MTE3_V` 让 V 队列等 MTE3 落地，
但它**不阻塞标量单元**。于是 pass p 的排水放行之后，标量单元立刻把 pass p+1 的三条
`DataCopyPad` 发进 MTE2 队列，MTE2 在 pass p 的 V 还在算归约时就把 `rvb` 覆盖成了
pass p+1 的 V。`PIPE_ALL` 之所以能盖住，是因为它是**标量级**的整机排水（连发射一起停），
和"MTE3 排空"根本不是一回事——这是 §11.12 记错的地方。

### 3. 落地：在读侧补一个自配对 V→MTE2 标记

最后一个落地缓冲读（rv 那次 `Cast`）之后插一条**自配对**的
`SetFlag<HardEvent::V_MTE2>(em2); WaitFlag<HardEvent::V_MTE2>(em2);`：
它后面的 DataCopyPad 必须等 V 队列走过这一点。自配对 = 每个 pass 一次 set 一次 wait、
不跨迭代带状态，所以没有 §11.12 里那种"需要预热、预热了还挂"的问题（`/tmp/pgq9.py`）。

| 变体（`/tmp/pgqK.py`，同进程交错 A/B） | pre_gram | 数值 |
|---|---:|---|
| HEAD（`PIPE_ALL`） | 3.916 | 基准 |
| 只有 `MTE3_V` 排水 | 3.426 | `Rv` 6136 坏 |
| `MTE3_V` 排水 + `V_MTE2` 标记 | **3.430** | **逐位一致** |

（对照组 `PIPE_ALL` + 三个 store-only 缓冲只验了数值：与"排水 + 标记"逐位同结果，
即 store-only 缓冲、`t2`/`gef` 复用本身没有问题，差异**只**在边界原语。）

标记本身 0.004 ms（噪声量级）：它只挡住**下一次装载的起步**，本遍 Gram 那半段
（`pga/pgk/pgb`）仍然和装载重叠。三个 store-only 的 bf16 缓冲（`qnb2`/`knb2`/`rvbo`）
是让两半互相独立的前提——没有它们，`MTE3` 还要读落地缓冲，读侧标记就盖不住写侧。

### 4. 验证与当前状态

- 逐位：`/tmp/pgqB.py`（2 chunk、交错 3 轮、13 个中间张量）`out-diff=0.000e+00`、
  `state-diff=0.000e+00`、`bad: clean`。
- 数值闸门：`out rel=8.620e-03 abs=5.490e-04 | state rel=4.245e-03 abs=2.453e-03`（未变）。
- `tests/test_persistent_loop.py`：6 项全过。

`[1,8192,96,128]`、CHUNK=64、MIN of 4：**8.459 ms**（pre_gram 3.442 / solve 2.534 / k2 2.477），
基线 8.960 → **−0.50 ms**。下一步仍然按老账本：pre_gram 剩的是 Gram 段的 Cube/V 配平，
solve 已到重叠地板，K2 还是 §11.10 的两次整机排水（0.65 ms）。

## 11.14. R3（七）：chunk 尾巴那 0.13 ms，用同一对标记拆掉（2026-09-14 深夜）

§11.13 把 pass 边界拆成"MTE3→V 排水 + V→MTE2 标记"之后，**chunk 尾巴的整机排水
就成了冗余**：它的两个活（读侧 WAR、写侧 WAR）正是最后一遍 pass 那对标记已经在管的。
唯一需要先调整的是 C=16：那里 NP=1、边界就是 chunk 边界，两个判断原来都带
`if (NP > 1)`，直接删尾巴排水会把两个 WAR 都露出来，所以**先把两对改成无条件**、
再删尾巴排水（C=64 的代码完全不变，只是编译期少一个分支）。

| 变体（`/tmp/r3a.py`，同进程交错，MIN of 3） | pre_gram | 数值 |
|---|---:|---|
| stock（含尾巴排水） | 3.441 | 基准 |
| 删尾巴排水 | **3.310** | 8 chunk × 3 轮、25 个中间张量全部逐位一致 |
| 删最后那个排水（每 block 一次） | 3.433 | 逐位一致，**没收益**（保留） |
| 两个都删 | 3.304 | 同上 |

验证：C=64 小形状 3 轮交错全逐位一致；C=16（16 chunk、3 轮交错、13 个中间张量）
`out-diff=0.000e+00`、`bad: clean`；数值闸门读数未变；`tests/test_persistent_loop.py`
6 项全过。

`[1,8192,96,128]`、CHUNK=64、MIN of 4：**8.317 ms**（pre_gram 3.304 / solve 2.534 / k2 2.463）。

### 剩下的账

- **k2 的两次整机排水 = 0.65 ms**，是现在最大的一笔。§11.10 的两次失败都用
  **跨迭代**事件对（会有配对漂移/挂死）；按 §11.13 的新认识，这里应该用
  **自配对**对：stage 2 开头（装载之前）一条 `SetFlag/WaitFlag`，让 MTE2 等
  "上一轮 stage 4 的 store 读完 UB"。但危险方向是 **MTE3-read → MTE2-write**，
  所以要的是 MTE3 产生的事件（不能拿 V→MTE2 顶替），这就是要查的下一个点。
- pre_gram 只剩"向量指令条数"这条线（Cube 免费、带宽免费、排水全清），
  要动只能减指令或加并行度。

## 11.15. R3（八）：k2 的两次排水现在**不要钱**了（0.65 → 0.00），以及一个探针方法论错误（2026-09-14 深夜）

§11.10 把 k2 的"每 chunk 两次整机排水"记成 0.65 ms，那是 **k2 还是 3.71 ms 的时候**
（§11.11 之前）。现在 k2 = 2.47，重新量：

| 变体（`/tmp/r3e.py`、`/tmp/r3f.py`，同进程交错、MIN of 3，**带 `A._defines()`**） | k2 | Δ | 数值 |
|---|---:|---:|---|
| HEAD（两次 `PipeBarrier<PIPE_ALL>`） | 2.484 | — | 基准 |
| stage 4 → 自配对 `HardEvent::MTE3_MTE2` | 2.471 | −0.013 | 逐位一致 |
| stage 2 + stage 4 都换 | 2.465 | −0.019 | 逐位一致 |
| **两次都直接删掉**（不安全） | 2.472 | **−0.012** | Vnew/VnewT 5.7 万点坏、d3 6 万点坏 |

**结论：这条 0.65 ms 的账已经不存在了**（§11.11 那次"一次 store 换 1.21 ms"把 k2 的 MTE3
形态整个换掉之后，排水等的那段已经被藏住了）。顺手确认了正确的原语：MTE3-read →
MTE2-write 要用 **`SetFlag/WaitFlag<HardEvent::MTE3_MTE2>`**（CANN 自己的 matmul 调度器
就是这么用的），而且必须**自配对、放在装载之前**——§11.10 里挂死的是跨迭代事件对。
改动本身只有 0.013 ms（噪声量级），所以**没有落库**，k2 那两行保持原样。

### 方法论错误（值得记一笔）

我这一轮的 k2 A/B 一开始把探针写成了
`rtc_compile(src.replace(名字), 新名字, "")`——**漏了 `A._defines()` 前缀**。
api 的 `_compile_all()` 是 `rtc_compile(head + source, name, "")`，那个 head 里有
`#define KDA_CHUNK 64`。少了它，`k2_persistent_loop.cpp` 就退回自己的 `#ifndef KDA_CHUNK 16`
——于是探针里跑的是 **M=16 的另一套 tiling**（而且 host 仍按 64 传 `nt`，算的东西根本不对），
k2 读数 1.45 ms vs 真实 2.47 ms。识别方法：**探针进程里 pre_gram/solve 与真实跑分对得上、
只有被改的那个 stage 对不上**——那就是编译配置不一致，不是优化生效。
（`/tmp/k2bar.py`、`/tmp/k2drain2.py`、`/tmp/k2min.py` 当年是带 `_defines()` 的，
所以 §11.10 的数字当时没错，只是**现在过时了**。）

当前：`[1,8192,96,128]`、CHUNK=64、MIN of 4：**8.317 ms**（pre_gram 3.304 / solve 2.534 / k2 2.463）。

## 11.16. R3（九）：pre_gram 的账本重做（3.31 → 3.27），以及三组旋钮的重扫（2026-09-14 深夜）

排水全清掉之后 pre_gram 的 3.31 ms 是"平"的，于是重做了一次"同进程内删一块、量一块"
（`/tmp/r3g.py`，C=64、`[1,8192,96,128]`、交错 MIN of 2，**基线 3.307**；这些变体结果本来就错，
只读时间）：

| 删掉 | pre_gram | Δ |
|---|---:|---:|
| 上一 chunk 的 `post_gram`（masking/取整/三张 store） | 3.007 | **−0.300** |
| gate cumsum（63 条串行 Add） | 3.082 | **−0.225** |
| 7 条早 store + Gram store（Qg/Kg/Rk/Rv/Ga/Gk/Gb） | 3.086 | **−0.221** |
| 每 pass 的 sigmoid 段（Muls/Exp/Adds/Dup/Div/Muls ×4） | 3.103 | **−0.204** |
| 两个 `RowReduce`（×4 pass） | 3.121 | **−0.186** |
| 三处 `Exp`（sigmoid + exp2(gate) + Gram 的两个） | 3.133 | **−0.175** |

单项都在 0.18–0.30 之间、加起来 1.31 ms——**没有任何一块占大头**，这就是"发射受限"的账本长相：
每条向量指令 ~27 cycle（§11.5），要快只能整体减指令。§11.12 里"Gram 半段 47%"那种结构性
大头在 mix 版里已经搬到 Cube 上了，剩下的全是小块。

旋钮重扫（同进程、交错）：

| 旋钮 | 结果 |
|---|---|
| `KDA_PRE_UNROLL`（每 block 的 chunk 数） | 8 → 3.518、16 → 3.374、32 → 3.315、**64 → 3.271**、96 → 3.771、128 → 3.508、192 → 4.872、256 → 3.416 |
| `KDA_SOLVE_OVERLAP` | 12 → 2.581、16 → 2.594、**24 → 2.505**、32 → 2.507、48 → 3.349 |
| `KDA_PERSIST_LOOP_BLOCKS`（k2） | 12/24 → 2.47、48 → 2.520、96 → 4.112 |

**落地**：`pre_unroll` 的公式从 `c // 384`（c=12288 时 32）改成 `c // 192`（=64），即
"固定 4 个 24-AIC wave、每 block 管 128 个 chunk"，数值不变（旋钮扫描里逐位一致）。
solve 的 overlap 24 和 k2 的 24 blocks 确认仍是最优，不动。

`[1,8192,96,128]`、CHUNK=64、MIN of 4：**8.279 ms**（pre_gram 3.275 / solve 2.524 / k2 2.480）。

## 11.17. R3（十）：这台机器已经打满了——"再排一排就能到 5 ms"这条路被证伪（2026-09-15 凌晨）

pre_gram 的账本变平（§11.16）之后，剩下的猜想是"三段的并行度不够，把阶段/分块流水起来
就能省"。**这个猜想被一个直接实验否掉了**（`/tmp/sat.py`）：

| 量法（同一进程，`[1,8192,*,128]`，CHUNK=64，三轮） | 时间 |
|---|---:|
| 一条 h=48 的完整流水 | 4.65–4.97 ms |
| 一条 h=96 的完整流水 | 8.27 ms |
| **两条 h=48 的完整流水、两个 stream 并发** | **8.93 ms** |

两条独立的半流水（总工作量 = 一条整流水）并发跑完是 8.93，比一条整流水的 8.27 只多 8%
——**说明这不是"延迟受限、靠并发能填"的形态，而是吞吐受限**：把同样的工作量怎么切、
放到几条 stream 上，都是同一个数。这条推论的推论是：

- 阶段间/分块间的软件流水、多 stream 切分、加大 grid 并发，**都不会带来收益**；
- 三段各自的账本（pre_gram 3.28 = 六个 0.18–0.30 的小块之和；solve 2.52；k2 2.46）
  就是这台机器做这件事的**吞吐地板**；
- pre_gram 里 Cube 是闲的（§11.12 量过：删掉 Cube 的活反而慢 0.30 ms），但剩下的
  AIV 活是逐元素的（norm/sigmoid/cumsum/exp2/mask），**没有 matmul 形状可以搬给 AIC**。

### 要到 5 ms 需要什么

按现在的算子清单，8.28 → 5.0 要砍掉 **~40% 的向量工作量**，只有三条路：

1. **元素级运算换 bf16**（向量单元每个 repeat 128 lane bf16 vs 64 lane fp32，理论 2×）。
   代价：破坏逐位一致，而且数值闸门余量已经很小（`out rel=8.620e-03`，容差就是 8.6e-3 量级），
   gate 的 cumsum/exp2 换 bf16 基本不可能过。粗估即使做成，也只到 6.0–6.5 ms。
2. **把还能搬的搬到 Cube**：只剩 gate cumsum（0.22 ms，三角 matmul 形状）这类零头，
   全搬也就 0.2–0.3 ms。
3. **改算法**：减少中间张量（例如把 qg/kg/rk/rv 的生成并进 Gram 的 operand 组装、
   或者直接用 KDA 的等价变形少算一个 pass），属于重写级别，且同样会动数值。

**结论：当前算法在这台 910B3 上的吞吐地板是 8.2–8.3 ms，而本轮 R3 已经从 8.96 走到 8.279**
（pre_gram 3.275 / solve 2.524 / k2 2.480）。5 ms 不是"再调一调"能到的，
需要一次"少算 40%"级别的算法或精度改动——那是一条要用户拍板的路线，不是排流水能解决的。

## 11.18. P0-1 + P0-2：公开 API 收口到 `persistent_loop`，历史 mode 移入 experimental（2026-09-15）

TODO 表的第一条是**正确性**问题，不是性能问题：`kda_bt16_fwd_ascendc` 的
`k2_mode` 默认值还是 `"separated"`，而 `separated` 那一串 kernel 是 **C=16 实现**
——它们的 `M` 不是 `KDA_CHUNK`，而是字面量 16：

| kernel | M 的来源 |
|---|---|
| `k2_d12.cpp` / `k2_vnew.cpp` / `k2_d34.cpp` / `k2_outstate*.cpp` | `constexpr int32_t M = 16` |
| `k2_mix_all_cube.cpp` / `k2_mix_d12_vnew.cpp` / `k2_mix_d4_outstate.cpp` | `constexpr int32_t M = 16` |
| `k2_persistent.cpp` / `k2_persistent_scan.cpp` / `k2_triton_aiv.cpp` | `M = 16` |
| `k2_d3_cube_bv64.cpp` / `k2_d4_full.cpp` / `k2_d4_only.cpp` | `M = 16, K = 16` |
| **`k2_persistent_loop.cpp`（唯一）** | **`#ifndef KDA_CHUNK` + `constexpr int32_t M = KDA_CHUNK`** |

所以在 CHUNK=64 的构建里走 `separated`，kernel 每 chunk 只读 16 行、只写 16 行：
**输出看起来合理、跑得飞快、但是错的**。§11.15 记的那次"1.45 ms 的优化"就是这个坑
（探针漏了 `_defines()`，kernel 退回默认 C=16）。默认值挂在这条路上，任何一个
"先不管 K2，把 K1 调好"的人都会踩到。

**改动**（`python/kda_ascendc_v1/api.py` + 新增 `experimental.py`）：

1. 公开入口 `kda_bt16_fwd_ascendc(..., k2_mode=None)`：`None` → `"persistent_loop"`，
   其余一律 `ValueError`（C=16-only 的 mode 报错文案指向 experimental）。
2. 实现体搬到私有 `_kda_fwd_impl`，公开/实验两个入口都走它；**C=16-only 的检查在
   实现体里**（`k2_mode in C16_ONLY_K2_MODES and CHUNK != 16` → 报错），所以没有任何
   调用路径能绕过它。
3. `C16_ONLY_K2_MODES` / `K2_MODES` / `PERSISTENT_LOOP` 成为模块常量，mode 名单只有
   一处定义。
4. 历史 mode 全部留在 `kda_ascendc_v1.experimental.kda_bt16_fwd_ascendc_experimental`，
   S12–S15 的对照测试和 `tools/bench_s*` 脚本改成 import 它（19 个文件，只改 import 行）；
   `benchmarks/bench_fla_compare.py` 按 mode 自动选入口，`--ascendc-modes` 两边的名字
   都能跑。

**验证**（两个 chunk 尺寸各一次全量进程）：

| 进程 | 内容 | 结果 |
|---|---|---|
| 默认（C=16） | `tests/test_api_k2_mode.py` + `tests/test_persistent_loop.py` | 22 passed，exit 0 |
| `KDA_CHUNK=64` | `tests/test_api_k2_mode.py` | 16 passed |

新测试 `tests/test_api_k2_mode.py` 钉住四件事：默认签名是 `None`（即推荐实现）、
公开入口对 10 个历史 mode 全部报错、未知 mode 报错、默认路径的 launch 账本是
"1 个 `kda_k2_persistent_loop` + 0 个 `kda_k2_d12_kernel` / `kda_k2_init_kernel` /
`kda_kg_transpose`"；并且在 C=16 构建下用 `persistent` 真跑一次、在 C=32/64 构建下
断言它按 `KDA_CHUNK=...` 报错。

**性能**：0 变化。生产路径本来就是 `persistent_loop`（`bench_fla_compare.py` 的默认
`--ascendc-modes` 就是它），本 commit 只把"别人也能踩到的那条路"关掉。
基线仍为 `[1,8192,96,128]`、CHUNK=64：**8.279 ms**（pre_gram 3.275 / solve 2.524 / k2 2.480）。

## 11.19. P0-3/P0-4/P0-5/P0-6：正确性与稳定性门禁，以及矩阵抓到的 C=32 是坏的（2026-09-15 上午）

这一轮不加性能改动，全部是"让错误答案无法静默通过"的机制。

### P0-5：所有 RTC 编译统一走 `_rtc()`，几何进 profile

`aclrtcCreateProg` 没有 `-D`，所以 `KDA_CHUNK` 等参数只能贴在源码前面（`_defines()`）。
漏贴不会编译报错——kernel 会退回自己的 `#ifndef KDA_CHUNK 16`，对着 C=64 的
host 调用只算 16 行（§11.15 那次"1.45 ms"的假优化）。现在：

- `api._rtc(rel, name)` 是**唯一**把源码交给编译器的入口（`_compile_all` /
  `_compile_persistent*` / `_compile_triton_aiv` 都改走它），
  `tests/test_d12_cube.py`、`tools/bench_d12_cube.py`、
  `tools/{compile,run}_cube_aic_probe_server.py` 这四处裸 `rtc_compile` 也改过来了；
- `tests/test_rtc_compile_config.py`（**纯 host，3 passed**）用 AST 扫描全仓库：
  任何地方出现直接 `rtc_compile(` 调用即失败；`kernels/v1/*.cpp` 里凡是读
  `KDA_CHUNK` 的文件都必须在 api.py 的编译表里出现，否则失败；
- `compile_config()` 把 `KDA_CHUNK / KDA_MAXH / KDA_SOLVE_WIDE_NCHUNK / SUPPORTED
  _SUBB / ASM_NCHUNK / WU_NCHUNK / SOLVE_OVERLAP` 写进每一份 `get_last_profile()`，
  并且测试断言它与 `_defines()` 逐项一致——测量和几何不再可能对不上。

### P0-3/P0-4：矩阵与稳定性门禁

- `tests/test_chunk_shape_matrix.py`：B∈{1,2} × H∈{2,32,48,96} × 短/长 T ×
  initial state 有/无，共 10 例，每例跑两次（determinism）+ 对 `test_torch_reference`
  的 fp32 参考比对（out/state 相对误差 < 2e-2）。chunk 是编译期常量，所以**每个
  build 跑一遍**：`bash tools/run_chunk_matrix.sh` 依次跑 C=16/32/64。
- `tests/test_stability_gate.py`：`[1,8192,96,128]` 连续 30 次（`KDA_STRESS_ITERS`
  可调）+ 两个 side shape 各 5 次，逐位一致 + 有限性；side shape 在循环之后还要过一遍
  fp32 参考，用来抓"单次调用没问题、设备被留在坏状态"的形态。

### 矩阵的结果：C=16 ✅ 13/13，C=64 ✅ 13/13，**C=32 ❌ 13/13**

C=32（本 commit 之前就存在，kernel 一个字节没动）的现象：

| 现象 | 数值 |
|---|---|
| 与 fp32 参考的相对误差 | out **1.27**、state **1.76**（单个 chunk 就已经错） |
| 同进程重复调用 | 第 2 次差 3.6e-3、第 3 次直接 **NaN** |
| `KDA_PRE_GRAM=aiv` | 同样 NaN（该路径在 C=64 本来就是坏的） |

逐段对照（host 复算 kernel 的公式，同进程、同一份输入）：

| 段 | C=64 | C=32 |
|---|---:|---:|
| gate cumsum + recentering + chunk 内 Gram（`Aqk`） | 1.3e-4 | **8.3e-5**（对） |
| solve：`A32` | 5.9e-2（两级 solve 的 bf16 耦合块） | 4.3e-4（对） |
| solve：`W` / `U` | 4.3e-4 / 1.9e-3 | **4.3e-4 / 1.3e-3**（对） |

也就是说 **K1 在 C=32 是干净的**（Gram、inverse、W/U 全部对上 host），故障在
`kernels/v1/k2_persistent_loop.cpp` 的 CHUNK=32 路径上（NaN + run-to-run 漂移 =
典型的边界/同步问题）。C=32 既不是默认（16）也不是生产配置（64），所以本轮的处置是
**收口而不是现场修**：

- `api.SUPPORTED_CHUNKS = {16, 64}`，`_kda_fwd_impl` 在**任何编译/下发之前**对
  `KDA_CHUNK=32` 直接报错，错误信息里写明现象、K1 已洗清、故障在 K2；
  `KDA_ALLOW_UNSUPPORTED_CHUNK=1` 留给后续调试用；
- 测试在这个 build 下变成"断言必须报错 + skip"，所以
  `bash tools/run_chunk_matrix.sh` 的 C=32 leg 输出
  `REFUSED`（不再伪装成 pass）；`tests/test_persistent_loop.py`、
  `tests/test_api_k2_mode.py` 的 device 用例同样加了模块级 skip；
- 记一笔待办：C=32 的 K2 需要的是一次 `ascendc-op-debug` 式的定位
  （K1 已排除、范围已缩到单个 kernel 的单个 chunk 尺寸）。

### P0-6：注释/文档对齐现状

- `api.py` 顶部 CHUNK 注释从"32 是更快的设置 / 16 是历史默认"改成现状：16 是默认
  （T%16 支持面最宽）、64 是全部 R3 数字和生产基线的配置（且 C>=64 才有两级 solve）、
  长序列请用 `KDA_CHUNK=64`；
- `docs/ASCENDC_V1_KERNELS.md`：K2 表头标明除 `k2_persistent_loop` 外全是 C=16-only
  kernel，只从 `experimental` 入口可达；verification 章节换成当前的命令与门禁说明。

### 本轮的验证

| 进程 | 内容 | 结果 |
|---|---|---|
| host | `test_rtc_compile_config.py` | 3 passed |
| host | `test_api_k2_mode.py -m "not npu"` | 13 passed |
| C=16 | matrix + stability + api + persistent_loop | 见上表 13/13 ✅ |
| C=32 | 同上 | 2 passed / 14 skipped（全部是"必须被拒绝"） |
| C=64 | 同上 | 13/13 ✅ |

性能：0 变化，基线**8.28 ms**（本 commit 不碰 kernel 与默认几何）。

## 11.20. P2-1 原型（一）：把 chunk 不变的工作搬出每 chunk 循环——以及 pre_gram 的真正瓶颈是"等 Cube"（2026-09-15 下午）

§11.16 的账本说 pre_gram 是"六个 0.18–0.30 的小块之和、发射受限"。本轮用同进程交错
A/B（`/tmp/pgqM.py` / `pgqN.py`，`[1,8192,96,128]`、CHUNK=64、MIN of 4–5）把其中
最大的一块（post_gram，0.300 ms）拆开量了：

| 变体 | pre_gram | 相对 | 数值 |
|---|---:|---:|---|
| ref（HEAD 的 kernel） | 3.576 / 3.652 | — | — |
| **P2-1：mask 位掩码提到 per-block** | **3.532 / 3.636** | **−0.044 / −0.016** | **逐位一致**（13 个中间张量 + out + state 全 0 diff） |
| 去掉 `CrossCoreWaitFlag(FL_DONE)`（结果错，只读时间） | 3.485 | **−0.167** | out-diff 8.1e-3 |
| 整段 post_gram 删掉（§11.16 的老数） | — | −0.300 | 错 |

也就是说 post_gram 的 0.30 ms 里：**~0.17 是等 Cube 的 flag，~0.13 才是 select/cast/store，
mask 那部分（每个 band 2 次 GM 读 + 2 次 Compare）只值 0.04**。§11.16 的"发射受限"结论
在这个位置上是错的：每 chunk 少 8 个 DataCopy + 128 条向量指令只换回 1.2%。

### 落地的改动（`kernels/v1/k1_pre_gram_mix.cpp`）

两个三角 mask 是**每 chunk 都一样**的 `[M, M]` 0/1 矩阵，而 select 只吃它的**位形式**。
于是位掩码改为**每 block 构建一次**（`bMfull`，`2 * M * M / 8 + 64` 字节，C=16 时 128 B、
C=64 时 1 KB），per-band 循环里直接按 `mbitsAll[mm * 16*M/8]` 取切片：

- 删掉：每 band 的 2 次 `DataCopy`（MaskS/MaskL，各 16×M fp32）+ 2 次 `Compares`；
- 保留：同样的 2 次 `Select`（bit-identical 的原因）；
- 顺带把每 chunk 32 KB 的 mask GM 读（12288 chunk × 32 KB = 393 MB/call）降到每 block 32 KB。

`bMbits`（旧的 per-band scratch）随之删除，UB 占用净减 512 B − 1 KB。

### 验证

- 交错 A/B 逐位一致（上面那张表）；refcmp 数值闸门**完全不变**：
  `out rel=8.620e-03 abs=5.490e-04 | state rel=4.245e-03 abs=2.453e-03`；
- e2e（CHUNK=64，MIN of 4）：**8.284 ms**，其中 pre_gram **3.251**（此前 3.275–3.283,
  这是 §11.16 以来的最好值）；
- C=16 矩阵 + 稳定性门禁复核（见 11.19 的 runner）。

### 这一轮的结论（对 5 ms 路线的影响）

pre_gram 剩下的账本是"每 chunk 的延迟链"：AIV 发完 chunk c 的 operand → 等 Cube 的
FL_DONE（**0.17 ms**，per-chunk ~1.3 µs 的实打实的 stall）→ post_gram。要拿这 0.17 只能
动那个 depth-one 的 flag 协议（"a subcore that runs a step ahead … the pairing drifts one
step per chunk"——代码注释里记的两次挂死就是它），风险极高、收益 2%；而**减指令**这条
路已经被本轮证伪（少 128 条/chunk 只换 1.2%）。所以 pre_gram 的可动空间只剩：

1. 协议加深（0.17 ms，有挂死风险，暂不动）；
2. 与 K1 下游合并（少一次 GM 往返级的工作，需要新的代数）。

## 11.21. P1-3：冻结"无收益"路线清单（2026-09-15 下午）

下面这些方向已经被**实测否掉**，除非有新证据，不要再开新的尝试（每一行都给出量法和数字，
以及它覆盖的范围）：

| 冻结的路线 | 证据 | 结论 |
|---|---|---|
| 加 stream / 多 stream 切分阶段 | §11.17 `/tmp/sat.py`：两条 h=48 半流水并发 8.93 ms vs 一条 h=96 的 8.27 ms | 吞吐受限，并发填不出收益 |
| 单纯加大 grid / 更多 block | §11.16 `KDA_PRE_UNROLL` 扫描（96→3.771、192→4.872） | 每多一个 block 就多付一次 prologue |
| 阶段间/分块间软件流水 | §11.17；以及 K2 的四段共享 staging（`ASCENDC_V1_KERNELS.md`） | 步与步之间是串行依赖 |
| 删 barrier / 减 descriptor / 相位粗化 | §11.13–11.14 已到极限；K2 的 drain 现在本来就免费（§11.15） | 剩下的 barrier 都是必需的 |
| **删"某一小块"来减 AIV 指令** | §11.20：每 chunk 少 8 个 DataCopy + 128 条向量指令 = **1.2%** | **本轮新增：指令数不是瓶颈** |
| gate cumsum 单独搬到 Cube（P2-4） | 同上：126 条 Add/chunk ≈ 1% 量级 | 上限从 0.2–0.3 ms 下修到 ~0.05，不值得单独做 |
| 选择性 bf16 降精度换吞吐 | §11.17：闸门余量只有 8.6e-3 对 2e-2 容差 | 风险/收益比差，留作最后一招 |
| C=32 构建 | §11.19：矩阵 13/13 全败（NaN + 漂移） | 已知坏，host 直接拒绝 |
| KDA_CHUNK=128 | UB 装不下（K2 的 4 头 state 已占 128 KB / 192 KB） | 用"更少的大 chunk"降步数这条路堵死 |

### 现在还剩什么（按证据强度排序）

1. **pre_gram 的 per-chunk 延迟链**：每 chunk-step 约 25 µs（3.25 ms / 128 chunk per block），
   其中被量到的"不隐藏"部分只有 ~1.4 µs（post_gram 0.30 + stores 0.22 + sigmoid 0.20 +
   RowReduce 0.19 + Exp 0.18 + wait 0.17 = 1.26 ms / 12288 chunk）。也就是说**大头是
   隐藏不掉的内存/握手延迟**（四次 256 B 粒度的 strided gather，外加 AIC 握手），
   不是任何一段算术。要动它只能改数据流（例如让 Cube 直接从公共布局取 operand、
   或把四次 gather 合并成一次），属于结构改动。
2. **K2 的步数与相位**：`k2_ms ≈ 512 步 × 4 相位 × nh × 0.65 µs`（`ASCENDC_V1_KERNELS.md`），
   步数是 T/CHUNK，相位是四个跨核 hop。要下来只能分段扫描（P2-5）或换代数（P2-2）。
3. **P2-2 代数重写**：唯一能一次性砍掉 20–40% 工作的路线；也是唯一有可能把
   `pre_gram + solve`（5.8 ms 中的大部分）从"每个 chunk 一遍 elementwise"里解放出来的办法。

### 本轮（P0/P1/P2-1）的净结果

- 公开 API 与几何收口（P0-1/P0-2/P0-5/P0-6）：默认走 `persistent_loop`、C=16-only kernel
  全部隔离并可报错、所有 RTC 编译强制带 defines、几何进 profile；
- 正确性/稳定性门禁（P0-3/P0-4）：10 例 × B/H/T/state 矩阵 + 30 次连续稳定性，
  一次跑出**C=32 是坏构建**这个既有 bug；
- 性能（P2-1）：pre_gram 3.275 → **3.251**，e2e **8.284 ms**（MIN of 4），数值闸门逐位不变。

## 11.22. P1-1：生产 benchmark 的 golden 与 gate——以及它抓到的 C=64 raw-gate 溢出（2026-09-15 下午）

目标（P1-1）：把 `[1,8192,96,128]`、C=64、`persistent_loop` 的生产数字**钉在仓库里**，
让"功能改好了但性能悄悄退 20%"变成一次非零退出，而不是一段没人复现的日志。

### 加了什么

| 部件 | 内容 |
|---|---|
| `benchmarks/golden/fla_compare_1_8192_96_128.json` | golden：median/p20/p80、stage 分解、first-call、几何（`compile`）、输入构造、FLA 对照、以及 gate 阈值 |
| `--update-golden` / `--gate` | 录制 / 检查；`--gate` 失败即 `exit 1`；`--kda-chunk` 决定构建几何 |
| `tools/run_bench_gate.sh` | 生产闸门一行命令（`--impl fla-alog,ascendc`，几何由 golden 定） |
| `tests/test_bench_gate.py` | 12 条 host 测试：20% 中位数回归、几何变化、数值漂移、stage 回归、NaN、缺参考各自必须失败，容差内的小幅变慢只能出 note |
| `tests/test_c64_gate_overflow.py` | 本轮新 fault 的最小复现（strict xfail） |

gate 检查四层，任何一层不过就 `FAIL`：**几何**（`compile` 逐键相等，RTC kernel 不带
编译期几何的自证，两个几何的耗时不可比）、**中位数**（golden×1.05 与绝对 8.5 ms 两条，
谁更严谁生效）、**p80**（×1.10，抖动）、**数值**（`o`/`state` 对 FLA 的 max-abs ≤ golden×1.5）。
stage 分解只做**归因**（×1.10 才报错），因为它是带 device sync 的 profile 数字。

关键取舍（都写进代码注释）：device 0 是共享的，同一构建在安静时 8.00 ms、别的租户忙时
能到 ~9.0 ms，所以中位数同时给"相对 golden 的比例"和"绝对天花板"两条，报告里点名是哪条
触发；而**几何不同一律硬失败**——"更快但是错的"是这个仓库已经发生过的失效模式
（C=16 kernel 回答 C=64 调用，1.45 ms）。

### 录制的第一个教训：几何必须显式

第一次录制我忘了 `KDA_CHUNK=64`，于是**录到了 C=16 构建的数字**：同一个 shape、
同一份输入，C=16 是 **11.565 ms**（pre_gram 3.549 / solve 2.129 / k2 **5.970**），
C=64 是 **8.003 ms**（pre_gram 3.226 / solve 2.508 / k2 **2.440**）—— 差的就是 K2 的步数
（T/CHUNK 从 128 变成 512）。所以 `--update-golden` 现在**拒绝**在没有 `--kda-chunk`
（或 `KDA_CHUNK`）的情况下录制，理由写进了拒绝消息。这条正好是 gate 自己存在的意义：
它检查的就是"数字和几何是不是一对"。（顺带一个可信度数据点：同一天先跑 `--impl ascendc`
再跑 `--kda-chunk 64`，两边的 `solve`/`pre_gram` 差 <2%，只有 chunk 是变量。）

### golden 的当前数字（2026-09-15 12:50 UTC，Ascend910_9382）

| 项 | 值 |
|---|---|
| ascendc `persistent_loop`，C=64 | **8.003 ms**（p20 7.993 / p80 8.015），first-call 48.3 s（RTC 编译） |
| stage（MIN of 4，`KDA_PROFILE=1`） | pre_gram 3.226 + solve 2.508 + k2 2.440 = 8.175 |
| 数值 vs FLA chunk64（A_log 配置） | `o` 9.766e-04 / `state` 4.745e-03 |
| FLA（本机，triton-ascend） | fla-bench chunk64 80.93 ms / fla-alog chunk64 80.96 ms |
| 比值 | 比 FLA 本机快 **10.08x**；比 FLA 公开的 H100 行（2.722 ms）慢 **2.95x** |
| gate 复核（新一轮进程） | 8.029 ms（+0.3%），stage 3.227/2.511/2.472，**PASS**，exit 0 |

### 它抓到的 fault：C=64 的 solve 在一个合法 raw gate 上溢出

golden 第一次录制出来的 `o-diff` 是 **nan**。追下去：

- 同一个进程里 `[1,8192,96,128]`、同一份输入：**C=16 有限（对 FLA 9.77e-4），C=64 全 NaN**；
- `return_intermediates` 定位：`Qn/Kn/Gate/Gc/Beta/Decay/Rk/Rv/Qg/Kg` **全部有限**，
  `Aqk32` 出现 `inf`（50.3M 项里 7402 个），下游 `L/W/U/out/state` 全是 NaN
  ——**溢出在 solve 自己组装 intra-chunk 矩阵的那一步**，不在它拿到的数据里
  （`k` 已 l2 归一化、`beta`/gate 都出自 sigmoid，量级都有界）；
- 最小复现只要 **T=128、H=2**（`tests/test_c64_gate_overflow.py`）：C=64 的 out NaN 16384 项、
  `Aqk32` inf 2 项；C=16 同样输入 0 项。C=16 是单个 16×16 solve，C=64 是两级块组装，
  这与 §11.5 记的"两级 solve"是同一条路；
- 触发条件是 **gate 的量程**，不是 shape：raw `g ~ N(0,1)` 经 `-5*sigmoid(exp(A_log)(g+dt_bias))`
  后有一半 token 顶到 −5 地板，chunk 内 `Gc` 走到 −320 量级；FLA 自己的 harness 生成的是
  `logsigmoid(...).clamp_min(-5)`（≈ −0.7/token），同一份张量 FLA 结果是有限的。
  实测四组（C=64，`[1,8192,96,128]`）：

  | gate 构造 | A_log | out |
  |---|---|---|
  | raw `N(0,1)` | `N(0,1)` | NaN 83.4M 项 |
  | raw `N(0,1)` | `linspace(-1,0.2)` | NaN 91.0M 项 |
  | raw `N(0,1)*0.1` | `linspace(-1,0.2)` | NaN 19.7M 项 |
  | `logsigmoid`（FLA harness 同款） | `N(0,1)` | **有限**（absmax 9.5e-2） |

处理方式（本轮）：
1. benchmark 的 `kda-model` 输入改用模型口径的 gate（`logsigmoid`），golden 的数值臂因此是活的，
   并在 golden 的 `inputs.note` 里写明；
2. 新 fault 用 **strict xfail** 钉住（C=64 必须 xfail、C=16 必须 pass，两腿一起构成对照），
   修好那天会变成 XPASS 强制更新；
3. gate 侧补硬规则：数值距离 **非有限** 就是 FAIL（`nan > x` 恒为 False，不显式判会**静默放过**
   一个 NaN 的运行——这正是本轮差点发生的事），golden 自己录到非有限值也要重录；
4. `tools/run_chunk_matrix.sh` 现在把这条对照跑进 C=16/C=32/C=64 三腿里。

### 根因（同日追到底）

`k1_pre_gram_mix.cpp` 的 `run_gram_aic` 用 Cube 直接算 `Aqk32 = ga @ gb^T`，
其中 `ga = q * exp(gc)`、`gb = k * exp(-gc)`（`Fixpipe` 那一行）——**衰减是折进操作数的**。
`gc` 是 chunk 内中心化的累加 gate，量程 = 该 chunk 的 gate 跨度：

- C=16：跨度上限 ≈ 16×5 = 80 → `exp(±40)` ≈ 2.4e17，fp32 装得下；
- C=64：跨度上限 ≈ 64×5 = 320 → `exp(±160)` = **inf**。

而**屏蔽在内的项（i ≥ j）需要两个操作数同时取到 inf 才能相乘**（早期 token 的 `exp(+gc)`
与晚期 token 的 `exp(-gc)` 都爆），乘积本身 `exp(gc_i - gc_j) ≤ 1` 却是可表示的。
所以这不是"结果溢出"，是**中间操作数溢出**；朴素地钳位两个指数会把乘积也钳坏
（e^80 · e^80 = e^160 仍然 inf），要保数值只能做**分子块重标定**：对角子块（16 行，跨度 ≤ 80）
继续用折叠形式，跨子块项直接用 `exp(gc_i - gc_j)`（≤1，天然安全）——这也是 flash-attention
那一类在线重标定的同款做法，属于结构性小改动，不是一行 clamp。

证据链（`T=128 H=2 C=64`，raw gate）：
`Aqk32` 的 inf 出现在**对角元**（`Aqk32[0,0] = -inf`，该 chunk 其余项全 0——`gc` 跨度 +116…-113
让屏蔽外的项全部下溢为 0），`L/W/U/out/state` 随之 NaN，而 `Qn/Kn/Gate/Gc/Beta/Decay/Rk/Rv/Qg/Kg`
全部有限。主机 fp64 复算：`exp(gc_i-gc_j)` 形式的 C 矩阵 absmax 0.87、逆 absmax 1.4e3（良态），
`exp(gc_j-gc_i)` 形式 absmax = inf —— 与上面的操作数分析一致。

**下一步**：这是一个**未修的正确性 fault**（优先级等同 P0-3 抓到的 C=32）。它不影响
真实模型口径的 gate（跨度 ~45，离 fp32 上限 10^15 远），但凡 gate 出现饱和的输入
（`|g|` 大、或未来换 `lower_bound`）就会从 C=64 开始给出 NaN 而不是"大而有限"的结果。
修法按上面的分子块重标定，开工前按 skill 的诊断协议走。

### P1-1 之后还剩什么

- **C=64 solve 溢出**（上面的新 fault）——正确性优先；
- **P2-2 代数重写**（唯一能一次砍 20–40% 工作的路线）；
- P1-2 形状矩阵、P1-4 workspace 精简、P2-3/P2-4/P2-5、P3-x；gate 本身是后续所有性能 PR 的入口。

## 11.23. 修 C=64 raw-gate 溢出：门控改按 32 行带宽参考，以及它暴露的一次 landing-buffer 竞争（2026-09-15 晚）

§11.22 把 C=64 的 NaN 定位到了"衰减折进 Gram 操作数"这一步，并留下一句"修法按分子块重标定"。
本轮把它修掉，`tests/test_c64_gate_overflow.py` 从 strict xfail 变成硬门禁。

### 改法：每个 32 行 gate 带一个参考行

`k1_pre_gram_mix.cpp` 的 `gc = gate - gate[mid]` 原来是**整 chunk 一个参考**（`mid = M/2`）。
31 个 5 的饱和门控能让一个 chunk 的 gate 跨度到 320（log2 域），两个操作数于是落到
`exp(±160)` = inf/0。改成**每个 32 行带一个参考**：

| 常量 | 值（C=64） | 含义 |
|---|---|---|
| `BS` | 32 | 一个 gate 带的行数（`M > 32 ? 32 : M`） |
| `MID0` | 16 | 带 0 的参考行（= 原 `M/2`，C≤32 时不变） |
| `XBAND` | `M > BS` | 是否存在第二个带；C=16/C=32 为 false，整段编译掉，**C=16 构建按位不变** |

参考行随 pass 走：`mid = MID0 + (hp * MT / BS) * BS`，C=64 的四个 16 行 pass 依次落在
16/16/48/48，即带 0（行 0–31）参考行 16、带 1（行 32–63）参考行 48。指数上限因此从
`exp(±160)` 降到 `exp(±80)` = 5.5e34，留 ~2000× 余量。

**跨带的那一块**：(1, 0) Gram 块的两个操作数来自不同带 —— i∈[32,64) 的 q 操作数按带 1 参考
（48），而 j∈[0,32) 的 k 操作数按带 0 参考（16），乘积会多出 `2^(R_j - R_i) ≠ 1` 的因子
（Gram 乘积的参考只有在两边**相同时**才相消）。这一块无法事后用标量修正，因为多余因子是
**逐 d** 的（gate 逐 d 不同）。所以带 0 的 k 侧**按带 1 的参考再发布一次**：

- AIV：两个带 0 的 pass（hp = 0/1）各自把自己的 16 行 `k * 2^-(g - g_48)` 写进
  `Gx[c * BS * D + hp * MT * D]`（一次 `Sub`/`Muls`/`Exp`/`Mul`/`Cast`，共 4 条 V 指令/2048 元素）；
- AIC：`run_gram_aic` 多收一个 `pGx`，把它作为 L0B 的第三个 tile（`BS × D`），在每步的
  s 循环里多算两个 `Mmad(BS, BS, D)`（`laX = la[2*DF*256]` = 带 1 的 A 操作数 × `lx`），
  用 `FixpipeParamsV220(BS, BS, BS, M, false)` 覆盖到 `Aqk32/L[m0 + BS * M]`——**覆盖**是安全的，
  因为上一次 Fixpipe 写的正是被拆开参考、值不对的那一块；
- `api.py`：`gram_x = torch.empty((c, max(1, CHUNK // 2), D), bf16)`（C≤32 时只占一行的位）。

(0, 1) 块（i∈[0,32), j∈[32,64)）不需要处理：它在三角掩码的**严格上三角**里，恒被清零。

数值（`/tmp/c64cmp.py`，fp64 复算，T=128 H=2 raw gate）：`A` 全块 rel 2.41e-3、
(1,0) 块 2.13e-2（该块是本征病态的，两个方向的指数在那里相消）、`L` 3.56e-3；
整条流水线 vs fp64 分块参考 **out rel 4.528e-3 abs 2.508e-4 / state rel 5.775e-3 abs 2.546e-3**。
`test_c64_gate_overflow.py` 另加一条断言：C>32 时 (1,0) 块必须非空（`> 1e-6`），
防止"把块悄悄清零"也能过。

### 验证时抓到的第二个 bug：新发布踩了 `rvbo` 的 WAR

C=64 全套测试第一轮 **4 条挂**（`persistent_loop` 的跨启动确定性、`[4-48]` 数值、稳定性 gate 两条），
且只在 pytest 里复现、单独跑就对。定位过程（每一步都留了探针）：

| 探针 | 做法 | 结论 |
|---|---|---|
| `/tmp/poison.py` | 把 `torch.empty` 出的 bf16 scratch 全填 NaN 再跑 | **不出现 NaN、结果正确** → 不是"读了没写的区域" |
| `/tmp/which2.py` | 同一进程里 plain（错）vs debug（对）逐张量比对 | `Rk/Rv/Qg/Kg/Beta/Decay/Ga/Gk/Gb` **按位相同**；`Aqk32`、`L`、`gram_x` 不同 → 坏在 Gram |
| `/tmp/gxval.py` | 拦 `torch.empty` 抓住 `gram_x` 逐块看 | 坏的那次 `max|Gx| = 2.7e-1`（正确是 8.8e-5），差值集中在**行 0–23**，量级正是 `v * beta` |

根因：新发布用 `rvbo`（本 pass 的 rv 落地区）当地板，而 pass 边界那对 `MTE3->V` 排水
**在这条 store 之前**，管不到它 —— 下一个 pass 的 `Cast(rvbo, ...)` 可以在 MTE3 读走之前
覆盖掉这块 tile（行 0–15 拿到下一个 pass 的 `v * beta`；行 16–23 是同一个竞争的另一半）。
修法是给这条 store **补一对自己的 `MTE3->V`**（`SetFlag/WaitFlag<HardEvent::MTE3_V>(e3x)`，
自配对的先例同 pass 边界的 `e3p`）。这条不是新代码"算错"，而是流水线排序漏了一条；
所有绕开它的写法（换落地区）都要么要新的 4 KB UB、要么把 WAR 挪到更近的下一个写者，
所以先按最贵的 0.075 ms 付账。

### 实测

| 项 | 结果 |
|---|---|
| 数值门禁（`/tmp/refcmp_full.py`，T=8192 H=8） | **未动**：out rel 8.620e-03 / state rel 4.245e-03（与 §11.22 前一致） |
| `KDA_CHUNK=64` 测试集（c64_gate + persistent_loop + stability + chunk_matrix） | 23 项全过（原 4 挂） |
| `KDA_CHUNK=16` 同集合 + `test_mix_aic_1_2` | 全过（C=16 路径未受影响，编译期整段消失） |
| 同进程交替 A/B（`/tmp/ab.py`，6 轮交替取 MIN） | pre_gram **3.367 → 3.442 ms（+0.075，+2.2%）**，solve/k2 不变，total **+0.062 ms（+0.75%）** |
| 生产 gate（`tools/run_bench_gate.sh`，C=64） | **PASS**：median 8.228 ms（golden 8.003×1.05 = 8.404，天花板 8.5），p80 8.239，`o` 9.766e-04，`state` 4.544e-03 |

A/B 数字说明一件事：**绝对数字不能跨进程比**。修完后第一次单独跑 `/tmp/final2.py` 得
pre_gram 3.447（看着像 +6%），同进程交替 A/B 才把它定到 +0.075（+2.2%）；其余那部分是
device 当前租户负载（同一轮里未修版本自己也从 3.24 涨到 3.37）。

可选的后续（**不阻塞**）：把那 0.075 ms 拿回来只要把落地区从 `rvbo` 换成一块**只由本段写**的
4 KB UB tile——那时相邻两次写之间恒隔着一次 pass 边界排水，`e3x` 就可以删掉；代价是 C=64
的 UB 余量（§11.12 记的"不到 8 KB"）可能不够，需要先量。

### 还剩什么

- ~~**C=32 `k2_persistent_loop`**：仍然坏（`api.SUPPORTED_CHUNKS` 里是已知不可用）~~ 已修，
  见 §11.24；
- `gram_x` 在 C=16/C=32 构建里仍占一行（`max(1, CHUNK // 2)`），未做 workspace 精简（P1-4）；
- P1-2 形状矩阵、P1-4 workspace、P2-2 代数重写（唯一能一次砍 20–40% 工作、通向 5 ms 的路线）。

## 11.24. 修 C=32：K2 stage 3 的跨带 B 操作数走错了带数（2026-09-17）

§11.19 把 C=32 的故障缩到 `k2_persistent_loop` 的 CHUNK=32 路径（NaN + run-to-run
漂移，K1 已逐个张量洗清），本轮就修它。

**定位**（`/tmp/c32.py`，`b=1 h=2 chunks=2` 带初始状态；每个张量查 finite，并跑两次比
determinism）：C=32 时从 `d1` 起 K2 的每张中间张量都带 NaN（`d1` 8192/16384、`d4`
32768/32768），而 K1 的 `Beta/Decay/Rk/Rv/Qg/Kg/Aqk/L/A16/A32/W/U` 全部 finite 且对上
host；同一个脚本在 C=64 干净。故障面是"整个 K2"而不是某一张输出 —— 共享的 L1 槽被写坏
的样子，不是某条算法算错。

**根因**：stage 3 的 `lx`（d34 的 B 操作数）装载。

```cpp
for (int32_t c = 0; c < BV / FR; ++c) {                  // ← 组数写成了 value 带数
    DataCopy(lx[iv * BV * K + c * (BV / FR) * FR * FR],
             Vt[t0 + c * FR * FR],
             DataCopyParams(BV / FR, FR * FR / 16,
                            (BV / FR) * FR * FR / 16 - FR * FR / 16, 0));  // ← 源跨距也按 4 算
}
```

`Vt` 是按 **A 操作数的块序**存的（`Vt[j0 * NB + m0]`，`NB = K / FR`，`m0` 是 chunk 行带、
`j0` 是 value 带），而 d34 的 B 操作数要的是 **[K, N] 的行主序分形** —— 和 A 操作数 [M, K]
的行主序是同一套约定，内层步长是 **n 带数 `BV / 16`（与 chunk 无关）**，所以两者之间是
**块转置**（`kb * (BV / FR) + nb`）而不是同一种下标写法。于是装载应当是：一次 burst 一个
16 x 16 块，**组走 `K / FR` 个 chunk 行带**，每组里 `BV / FR` 个 burst 沿源跨距
`K / FR` 个分形跳。代码把三处 `K / FR` 都写成了 `BV / FR`：

- C=64（K = 64）时 `K / FR = BV / FR = 4` —— **逐字相同**，这就是它一路过门禁的原因；
- C=16 走上面的 `K == FR` 直读分支（`NB = 1`，两种写法都退化成 identity），也看不见；
- C=32（`K / FR = 2`）时：每组 4 个 burst、源跨距按 4 个分形算 → 一次 iv 迭代写
  `4 x 4 x 256 x 2 B = 8 KB`，而 `qx` 的一个 iv 槽只有 `BV * K * 2 = 4 KB`。第一次迭代把
  两个槽一起写掉，第二次**越过 `qx` 末尾 4 KB**，落在 L1 arena 里紧跟 `qx` 的 `qk`
  （`kg^T`，8 KB）上。stage 1/3 共享这块 arena，所以 K2 的中间张量（`d1` 起）一起 NaN；
  AIV stage 2 那两条 `PipeBarrier<PIPE_ALL>` 决定越界写与后续读的先后，故障是否显形就
  随运行而变 —— 与观测到的 run-to-run 漂移一致。

**修法**（`kernels/v1/k2_persistent_loop.cpp`，两处 token）：组数 `BV / FR` → `K / FR`，
源跨距 `(BV / FR) * FR * FR / 16 - FR * FR / 16` → `(K / FR) * FR * FR / 16 - FR * FR / 16`。
C=64 时两式同值 → 生产路径逐位不变；C=16 走直读分支 → 也不动。

**实测**：

| 项 | 结果 |
|---|---|
| 数值门禁（`/tmp/refcmp_full.py`，T=8192 H=8） | C=32 **与 C=64 逐位同数**：out rel 8.620e-03 abs 5.490e-04 / state rel 4.245e-03 abs 2.453e-03（chunk 不变性直接体现） |
| `bash tools/run_chunk_matrix.sh` | C=16 / C=32 / C=64 三条腿全 **PASS**（C=32 从 `REFUSED` 变成真 pass） |
| `pytest tests/`（全目录） | C=32 全过、C=64 全过；C=16 只剩 3 条 `test_persistent_scan.py`（既有问题：`k2_persistent_scan.cpp` 在当前 RTC 工具链下编不过，与本轮无关） |
| 生产 gate（`tools/run_bench_gate.sh`，C=64） | **PASS**：median 8.235 ms（golden 8.003 x 1.05），`o` 9.766e-04 / `state` 4.544e-03 与 golden 一致 |
| 端到端计时（`/tmp/final2.py`，MIN of 4，同机交替两轮，[1,8192,96,128]） | C=64 **8.477 / 8.464 ms**（pre_gram 3.445 / solve 2.548 / k2 2.462）；C=32 **8.905 / 8.905 ms**（pre_gram 3.189 / solve 2.294 / k2 3.405） |

C=32 的取舍：K1 反而快 0.5 ms（pre_gram -0.26、solve -0.25，chunk 数翻倍换来更小的
Gram/solve 块），K2 慢 0.94 ms（256 个 chunk 的 flag 与描述符账），合计 **+0.43 ms
（+5%）**。**生产构建仍是 C=64**；C=32 现在的价值是"第三方形状需要更细的 chunk 时它可用"，
以及 chunk 不变性多了一个非平凡验证点（C=16 与 C=64 恰好都无法验证 stage 3 的跨带走动）。

**API/文档收口**：`SUPPORTED_CHUNKS = {16, 32, 64}`；拒绝文案从"C=32 已知坏"改成
"未验证的 chunk 尺寸"（并举例：C=128 会把 L0C 队列与 L0A 分配各顶到 128 KB / 64 KB 的
整块上限，而这类错误只会在核侧以 aivec error 或静默错值出现，host 不会报）；`tests/test_persistent_scan*.py` 与
`test_output_layout.py`（钉的都是 C=16-only 的实验 mode）、`test_mix_aic_1_2.py` /
`test_s15_mix_d12_vnew.py`（T = 16 的用例）在非 C=16 构建下改成模块级 skip —— 前两个
之前是**收集期报错**，`pytest tests/` 在 C=32/64 下根本跑不起来；三个断言"不支持构建必须
报错"的测试的 `match=` 文案同步更新；`tools/run_chunk_matrix.sh` 的 C=32 leg 不再打印
`REFUSED`。

### 还剩什么

- C=32 的 K2 比 C=64 慢 0.94 ms（stage 3 现在 2 组各一次 DataCopy、C=64 是 4 组）：
  若以后要常驻 C=32，值得按 §11.6.5 的描述符账再收一遍；
- `gram_x` 在 C=16/C=32 构建里仍占一行（`max(1, CHUNK // 2)`），未做 workspace 精简（P1-4）；
- P1-2 形状矩阵、P1-4 workspace、P2-2 代数重写（唯一能一次砍 20–40% 工作、通向 5 ms 的路线）。
