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
