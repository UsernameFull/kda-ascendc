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
| **删"某一小块"来减 AIV 指令** | §11.20：每 chunk 少 8 个 DataCopy + 128 条向量指令 = **1.2%** | **本轮新增：指令数不是瓶颈**；**§11.55 重开**：加/删双向实测 31.7 ns/条、删块 −0.3 ms/块，1.2% 是删除点落在 FL_DONE 到达上的个例 |
| gate cumsum 单独搬到 Cube（P2-4） | 同上：126 条 Add/chunk ≈ 1% 量级 | 上限从 0.2–0.3 ms 下修到 ~0.05，不值得单独做；**§11.55 修订**：删 cumsum 实测 −0.3 ms，搬到 Cube 仍不值（GM 往返 ~0.4 ms），但 AIV 内的无冒险分块扫描预估净得 0.15–0.2 ms；**§11.56 已落生产**：设备 −0.115 ms、e2e 配对 −0.054 ~ −0.096 ms |
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
| `pytest tests/`（全目录） | C=16 / C=32 / C=64 全过（C=16 那 3 条 `test_persistent_scan.py` 是 `k2_persistent_scan.cpp` 的 UTF-8 BOM 造成的 RTC 编译失败，已在后续修复中清零：v1 源码 BOM 剥离 + `api._rtc` 按 `utf-8-sig` 读取） |
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

## 11.25. 吞吐侧第二轮：三条候选的实测判决，以及一处落地（2026-09-21）

§11.21 冻结的清单之后，唯一没被实测否掉的吞吐方向是"少算工作"（P2-2 代数重写）。本轮
把它连同两条相邻路线一起量了，并落地了其中唯一一条干净收益。所有数字都是**同进程交错
A/B**（K2 单独计时 = 直接重放捕获的 launch args；pre_gram 单独计时 = 同法），因为本机
绝对值受设备影响（见 §11.26）。

### 1. P2-2 的数学成立，但收益来源是错的（判决：冻结）

**代数复核**（`/tmp/p22_state.py`、`/tmp/p22_shallow.py`）。把 `v_new` 代入它的两个消费者：

```
o      = scale*(qg @ S^T) + Aqk @ v_new   = R @ S^T + Ao
S_new  = S*d + v_new^T @ kg               = S*d - S @ G + Ukg
         R = scale*qg - Aqk@W   Ao = Aqk@u   G = W^T@kg   Ukg = u^T@kg
```

fp64 恒等式残差 6.245e-17，四个 chunk 常量按 bf16 存储后的数值：

| gate 深度 | max\|G\| | max\|W\| | out_rel | state_rel | 判定（闸门 2e-2） |
|---|---:|---:|---:|---:|---|
| near-zero gate | 4.847e-01 | 2.962e-01 | 2.391e-03 | 2.122e-03 | PASS |
| 浅 gate (LB=-0.5) | 4.151e-12 | 1.645e-01 | 2.505e-03 | 2.907e-03 | PASS |
| 生产 (LB=-5) | 2.619e-18 | 1.377e-01 | 2.541e-03 | 2.422e-03 | PASS |

深度扫描是**必要的**：第一版只在生产 gate 上跑，`max|G| = 1.6e-16` 说明 `-S@G` 项
根本没被走到——那是假阳性。补浅 gate 才确认 G 为 O(1) 时也成立。

**收益来源被证伪**（`/tmp/k2hop.py`、`/tmp/k2hop2.py`、`/tmp/k2hop3.py`）。文档 §11.7 的
`k2_ms ≈ 512 steps x 4 phases x nh x 0.65 us` 模型暗示"四跳降到两跳"能省一半协议时间。
同进程交错、MIN of 11：

| 变体 | min_ms | Δ | 说明 |
|---|---:|---:|---|
| stock | 4.090 | — | 512 head-steps，7.99 µs/step |
| V0 stock 重新编译 | 4.101 | +0.012 | **噪声底 ±0.014** |
| V1 2-hop（v_new 折叠后的握手） | 4.050 | **−0.039** | 不是一半协议时间 |
| V3 所有跨核 flag 全删 | 4.404 | **+0.314** | 反而更慢 |
| V4 删两个 v_new store | 3.967 | −0.123 | |
| V4b 删 store + 16 次 Transpose | 3.877 | **−0.214** | 结构地板 |

两条读数：**V3 是决定性的**——把跨核 flag 全删掉 K2 反而慢 0.314 ms，与 §11.7 那句
"handshakes coarsened 16/step → 4/step = 12.14（更慢）"同现象：这些 flag 不是同步开销，
而是**给两个引擎让出节奏**。所以 hop 数不是可优化维度，§11.7 的相位模型在 2.46 ms 形态下
已经失效。**P2-2 的代价仍然是 +0.3 ms 流量（G+Ukg 各 32 KB/chunk-head = 786 MB 往返），
净收益为负。**

### 2. 方案 C（host 侧预排 chunk 连续布局）：负的（判决：冻结）

思路：把 q/k/v/g 在 host 端 permute 成 chunk 连续，让 kernel 的 gather 变成连续读。先量
**kernel 侧上界**——把四次 `DataCopyPad` 的行间距改成 0（数值会错，只读时间）：

| | min_ms | median |
|---|---:|---:|
| stock | 4.182 | 4.191 |
| 连续读 | 4.160 | 4.178 |
| **差值** | **−0.022** | ← 噪声量级 |

host 侧预排实测 **2.104 ms**（2.01 GB 搬运）。**净 −2.082 ms。** 与 §11.12 的
"1 GB 流量只值 0.2 ms"完全一致：C=64 时读全在 L2，gather 形态无所谓。

### 3. pre_gram 的 pass 边界：找到 0.230 ms，但收不了（判决：冻结）

§11.12 把 pass 边界的整机排水记成 0.54 ms；§11.13 用两对自配对事件 marker 替掉了它。
本轮量的是**替掉之后还剩多少**（`/tmp/passmark.py`，MIN of 11）：

| 变体 | min_ms | Δ |
|---|---:|---:|
| stock | 4.171 | — |
| V0 重新编译 | 4.165 | −0.006（噪声底） |
| D1 删 load 侧 `V_MTE2` marker | 4.150 | **−0.021** |
| **D2 删 store 侧 `MTE3_V` marker** | **3.941** | **−0.230** |
| D3 删 `FL_DONE` 等待 | 4.157 | −0.014 |
| D4 两个 marker 都删 | 3.935 | −0.236 |

**D4 ≈ D2** → load 侧那对已经是死的。**D3 是过期数字**：§11.20 记的 0.167 ms 现在只值
0.014，后续改动早已把它盖住，该条从待办划掉。

**D2 的 0.230 ms 收不了**（`/tmp/d2num.py`）。按仓库规矩"先看逐位一致，再看时间"，
删掉 marker 后跑全流程 diff 全部中间张量：

```
Rv 3.145e-01   U 3.203e-01   Vnew/VnewT 3.203e-01   out 5.830e-03
A16/A32 4.932e-02   L 4.949e-02   Aqk 1.364e-02   W 1.4e-15
```

**marker 是负载承载的。** 安全收掉它需要把跨 pass 复用的 4 个 tile
（`qgb/kgb/rkb/rvbo`）双缓冲 = **16 KB**，而实测 UB 余量只有 **4 KB**：

| 量法 | 结果 |
|---|---|
| 新加 1 KB dummy | FAIL（新分配有额外对齐开销） |
| 扩大已有活 buffer（`bEf`） | +4 KB OK，**+5 KB FAIL** |
| 名义账（逐个 `InitBuffer` 求和） | 185.34 KB / 192 → 6.66 KB |

唯一能腾 16 KB 的大块是 `bT0`（整 chunk gate，32 KB），但它**切不了**：`gf` 的读者里
cumsum 需要整 chunk 的顺序链、`kg` 需要第 `M-1` 行、`XBAND` 需要第 `MID0+BS` 行。按 band
切就得每 band 重跑 cumsum，而 cumsum 实测 0.225 ms → **净收益 ≈ 0**，还引入跨 band 进位
的正确性风险。

### 4. 落地：`V` 死 store 的守卫

`k2_persistent_loop.cpp` 每个 chunk-head 写**两份** `v_new`：`V`（row-major `[M, BV]`）
和 `Vt`（packed 块序）。grep 证实**这份 kernel 从不读 `V`**——AIC 的每个操作数都取自
`Vt`（stage 1/3），唯一的消费者是 host，且只在 `return_intermediates` 下。所以按
`k1_pre_gram_mix.cpp` 的 `pQn/pKn` 模式加守卫：

```cpp
if (pVnew != nullptr) DataCopy(V[out0], vb, DataCopyParams(M, BV / 16, 0, 0));
```

`api.py` 生产路径传 `nullptr`，只在 `return_intermediates` 时分配（顺带省掉 201 MB 分配）。

| 量法 | old | new | Δ |
|---|---:|---:|---:|
| K2 单独（捕获 args 重放，MIN of 11 交错） | 4.088 | 4.017 | **−0.070** |
| 端到端（同进程交错，MIN of 9） | 14.583 | 14.467 | **−0.117** |

逐位一致（`return_intermediates` 下 out/state diff = 0.000e+00，Vnew 仍可读出）；
`bash tools/run_chunk_matrix.sh` C=16/32/64 三腿全 **PASS**；
`test_persistent_loop.py` / `test_c64_gate_overflow.py` / `test_output_layout.py` /
`test_api_k2_mode.py` 全 PASS。

### 5. 本轮之后还剩什么

| 方向 | 上界 | 状态 |
|---|---:|---|
| P2-2 代数重写 | 相位零收益，代价 +0.3 ms | **冻结** |
| 方案 C host 预排 | −2.082 ms（负） | **冻结** |
| K1 pass 双缓冲 | 0.230，需 16 KB | **冻结**（UB 只有 4 KB，腾空间的代价 ≈ 收益） |
| K1 更多工作搬给空闲 Cube | 0.05–0.3 | 只剩零头（gate cumsum ~0.05） |
| solve 重叠深度 | 0.6 | 只能靠 slice 粒度重构 |

到 5 ms 量级仍然需要一次"少算 40%"级别的算法或精度改动（§11.17 的结论未变），而 P2-2
这条唯一的候选路线已在本轮被证伪。

## 11.26. 本机绝对时间为什么与 golden 差 1.35x——以及一个测量口径的坑（2026-09-21）

排查一次"端到端 14 ms vs golden 8.0 ms"的表象，结论：**三个因素叠加，前两个是口径问题。**

**① 探针的 `KDA_PROFILE=1` 代价 +2.94 ms。** 它每个 stage 边界做一次
`torch.npu.synchronize()`，阻断 host 对下一段的预投：

```
KDA_PROFILE=1   wall 14.179   (pre_gram 4.303 + solve 4.924 + k2 4.019 = 13.246)
KDA_PROFILE=0   wall 11.240
差               +2.939 ms
```

**② 单调用 `perf_counter` 取 MIN 不等于仓库口径。** 生产 benchmark 用
`triton.testing.do_bench`（warmup 100 ms / rep 1000 ms，报中位数与 p20/p80）。
`bench_fla_compare.py --gate` 同机同 build 读数 **10.773 ms（p20 10.745 / p80 10.808）**
——p20 与 p80 只差 0.06 ms，说明这一次测量内部很稳，10.773 是可信的当前值。

**③ 设备不同。** golden 记录于 `Ascend910_9382`，现在跑在 `Ascend910B3`；bench 自己
就会打印 `note: device Ascend910B3 != golden's Ascend910_9382 (timings are
device-specific)`。本机负载在 9–183 之间波动，同一天同一份代码量到过 18.9 ms
（`/tmp/segA_hsweep.log`）和 10.8 ms。

**口径规则（写给未来的自己）**：

- 与 golden 比绝对时间只在**同设备**下有意义；跨设备只能比同一格式内的比值；
- 报"优化了多少"必须用**同进程交错 A/B**，噪声底用"stock 重新编译一遍"标定
  （本轮 K2 的底是 ±0.014 ms，pre_gram 是 ±0.006 ms）；
- 阶段计时一律带 `KDA_PROFILE=1` 说明，别把它当端到端时间报。

**分段对照**（golden vs 本机同口径，比值仅供参考）：

| stage | golden | now | 比值 |
|---|---:|---:|---:|
| pre_gram | 3.226 | 4.300 | 1.33x |
| solve | 2.508 | 5.057 | 2.02x |
| k2 | 2.440 | 4.082 | 1.67x |
| total | 8.003 | 10.773 | 1.35x |

`solve` 的 2.02x 与已落地的改动无关（本轮只动了 K2 的 `V` store，solve 不碰那份 kernel），
且同一天的日志里 solve 量到过 2.495 ms（与 golden 的 2.508 几乎一致）——solve 靠 AIV/AIC
双流重叠，负载高时重叠最先受损，符合"设备/负载差异"而非回归。

## 11.27. 跨阶段并发实测：可隐藏的空间在这台机器上已经用掉了（2026-09-21）

一条被反复提起的重构提案：把 `pre_gram → solve → K2` 改写成按 tile/head-group 流式的
**persistent super-kernel**，用阶段重叠 + 中间张量不落 GM + 统一布局换几毫秒。这个提案
的前提是"阶段之间可并发"。§11.17 否掉过"阶段/分块流水"，但那条证据只能覆盖它测过的形态：

**§11.17 测的是"两条相同的半流水"**（`/tmp/sat.py`：两条 h=48 并发 8.93 ms vs 一条
h=96 的 8.27 ms）。它证明的是**同一个 grid 已经打满吞吐**——把同样的工作怎么切都一个数。
它**不能**回答"两个**不同**阶段能不能互补"，而三段的资源画像确实不同：pre_gram 是 AIV
向量受限（§11.16，Cube 闲）、solve 的 AIC 半边只吃 Cube（§11.9）、K2 是等 flag 受限
（§11.15，Cube 忙）。所以这条前提必须单独量。

### 量法

`tools/probe_stage_overlap.py`：捕获生产调用的 launch 序列（74 个 launch）后按 kernel 名
重放，每个 arm 都是**真 kernel、真参数、真 grid**，只改 stream 归属；并发 arm 共用张量，
输出无意义（仓库既有的"删一块/只读时钟"口径）。同一进程、MIN of 7，`[1,8192,96,128]`、
C=64。四次独立运行的读数一致（±0.05 ms）。

### 结果（ms）

| arm | 隔离 | 并发 | 隐藏 |
|---|---:|---:|---:|
| pre_gram | 4.22 | | |
| solve:AIV-wide | 2.40 | | |
| solve:AIC（assemble+cube） | 2.59 | | |
| k2 | 4.06 | | |
| **stage sum** | **13.27** | | |
| PG ‖ K2 | | 7.58 | 0.70（8.4%）|
| PG ‖ solve:AIC | | 6.64 | 0.17（2.5%）|
| **solve:AIV ‖ K2** | | **4.75** | **1.71（26.4%）**|
| PG ‖ solve:AIV | | 6.39 | 0.22（3.3%）|
| K2 ‖ K2（对照） | | 6.26 | 1.86（22.9%）|

读法：

- **`solve:AIV ‖ K2 = 1.71 ms` 是唯一的大块**，而它的机制不是"阶段不同"，而是
  **solve 自己就有 AIV/AIC 双流重叠**（§11.9：AIV 1.84 + AIC 1.28 重叠到 2.53）——
  solve:AIV 那 2.40 ms 期间 Cube 是**闲**的，K2 正好把它吃掉。也就是说这 1.71 ms 属于
  **solve 已经吃掉的那 0.6 ms 的同一份预算**，不是新的空间；
- `PG ‖ K2 = 0.70 ms`：K2 的控制流里本来就有等 AIV/Cube 的空档，pre_gram 能填进去一点，
  与 §11.7/§11.15 的"K2 的跨核 flag 是节奏让位、不是同步开销"一致；
- `PG ‖ solve:AIV = 0.22`、`PG ‖ solve:AIC = 0.17`：都是噪声量级——**两个 AIV 受限的
  阶段并联没有收益**（这正是 §11.17 覆盖的形态）；
- `K2 ‖ K2 = 1.86` 是负结果里最有意思的一条：K2 自己并发两份能隐藏 23%，说明它是**延迟
  受限**（24 块、每块一串 512 步的依赖链），但**这条窗口已经被生产形态用掉了**——生产
  K2 已经是 24 块铺满 24 个 AIC 的一波。

### 判决：冻结整条路线

把上表所有**在当前形态下还没被吃掉的**空间加起来，乐观上界是 0.70（PG‖K2）+ 0.22 + 0.17
≈ **1.1 ms**，而且这 1.1 ms 已经假设了"tile 依赖全部免费、UB 全部够、调度零成本"。
对照提案自己的账（预期 6.5–8.0 ms，即 2.8–4.3 ms 的收益），**差 3–4 倍**，而代价是：

- 三段的 chunk 循环互相嵌套（K2 的步内依赖是 512 步串行，§11.7），做成一个常驻 kernel
  要把 pre_gram 的 per-chunk ~450 条 AIV 指令塞进 K2 的每 step（§T2.3 已判：+3 ms）；
- 中间张量不落 GM 只能省 0.1–0.25 ms 量级：§11.12 实测 **1 GB 流量在 C=64 时只值
  0.2 ms**（全在 L2），而提案假设的 0.5–1.2 ms 是按 HBM 带宽算的；
- K2 的 launch 已经是 1 次、pre_gram 1 次（74 个 launch 里 72 个是 solve 的切片对），
  "消除 host launch/同步"的上界 = **host-only 实测 4.7 ms 是打桩后的纯 host 时间**，
  但这份时间在真机上**与 device 时间是重叠的**（否则 11.26 的 wall 不会是 10.8 而不是
  13.2+4.7）——真正暴露的 host 份额是 11.26 量的 **0.873 ms**，且其中主要是每调用一次
  的 workspace 分配（~0.67 GB）与参数打包，不是 launch 次数。

**净结论：这条路线的收益上界 ~1.1 ms（且主要落在已被 solve 吃掉的预算里），成本是
三段合并的重写。判决：冻结。** §11.17 的吞吐结论在"跨阶段"这个维度上也不成立为
"可优化空间"——不是因为没有并发窗口，而是因为**这台机器的并发窗口在生产形态下已经
基本用掉了**：能重叠的都已经重叠（solve 的双流、K2 的 24 块单波），剩下的窗口加起来
1 ms 量级。

### 对"还能往哪走"的更新

§11.25 的清单不变，仍然只有"少算工作"（算法/精度）这一条能到 5 ms 量级；本轮的增量是：
**"排流水"这条在 11.17 之后被再次确认封死，而且这次是在更有利的测法下**（跨阶段、
真 kernel、真 grid）。新增一条负结果，避免以后再花一轮去试 super-kernel。

## 11.28. 全流水生命周期重构（一）：S1 冻结、三段账本落地，以及它抓到的三处 debug-only 写（2026-09-22）

这一轮的提案是"把 `pre_gram -> solve -> K2` 按生命周期重写：减少中间张量落 GM、扩大阶段重叠"。
提案的前提是 §11.27 已经量过的那张表：隔离总和 13.27 ms，**在当前形态下还没被吃掉的**
重叠上界约 1.1 ms（PG‖K2 0.70 + PG‖solve:AIV 0.22 + PG‖solve:AIC 0.17）。所以这一轮不再
论证"能不能重叠"，而是先把三段的生命周期与字节账做成**可复算的产物**，再用它给候选排序。
设计书见 `docs/PREFILL_LIFECYCLE_REFACTOR_20260922.md`。

### 1. S1（workspace 池化）：三条证据，判决冻结

§11.26 把 0.873 ms 的暴露 host 份额指向"每次调用的 workspace 分配"，S1 就是把这 28 个
张量、4.15 GB 的分配缓存到按 shape 键控的池里。三份证据把它否掉：

| 证据 | 读数 | 结论 |
|---|---|---|
| `tools/probe_host_cost.py` | 分配臂单独 0.304 ms/调用（host MIN of 12） | 池化能省的**全部**就是这个数 |
| `tools/probe_workspace_pool.py` | checkout/checkin 0.00x ms，池化后仍持有 4.15 GB | 手臂真实，但只值 0.30 ms |
| `tools/probe_host_exposure.py` | e2e 中位 10.773 ms 在 host wall 5.84 → 9.34 → 10.90 ms 上**保持平**（< +0.05），到 13.63/18.32 ms 才跟涨（+3.1/+7.6） | host slack ≥ 5 ms，0.30 ms 埋在 slack 里 |

第三条是决定性的：**host 侧能白扛一个设备时间的量级**，所以"省 0.30 ms host"不可能出现在
端到端数上。同轮还量到池化原型的两种错法（都属于"更快但答案是错的"这一类）：

- 按 `(shape, dtype)` 键控会把 `rk/rv/qg/kg/W/U` 这些同形 buffer 相互别名（实测
  `max|do| 3.368e-01`、`max|ds| 2.799e-01`，而 e2e 看起来还"快了" 0.606 ms）；
- 交给上层一个"够形状不够元素数"的 buffer，会撞上 api 自己的 `.view()`（`shape [8,2,64,128]
  is invalid for input of size 262144`），或在更大的请求上越界读。

负结果被两个文件钉住：`tests/test_workspace_pool.py`（容器契约：互不别名、忙则拒绝、
换 shape 换条目、buffer 够大）与 `tests/test_workspace_pool_numerics.py`（暴露口径：
加 host 延迟看 e2e 是否跟涨；host-bound 的小形状必须一对一跟涨）。**S1 冻结。**

### 2. Level 0：三段生命周期 / slot / 流量账

新增 `tools/gen_stage_lifecycle.py`，产出三份产物（`docs/artifacts/`）：

| 产物 | 内容 |
|---|---|
| `stage_lifecycle.csv` | 44 行 8 字段账本：producer / consumer / scope / memory / lifetime / sync / layout / role，覆盖 host、pre_gram、solve(wide/assemble/cube)、K2 |
| `stage_slot_map.csv` | 跨 stage 交接与双 window ring 的 local slot 分配（`pre_gram->solve` 2 槽、`solve->k2` 1 槽、`pre_gram->k2` 3 槽） |
| `stage_traffic.txt` | 每次调用的 GM 读写账，按 stage 与 producer/consumer 拆分 |

账本不是注释：**每一行都要通过引用校验**——行内引用的符号必须出现在它引用的 kernel（或
`api.py`）里，交接点引用的代码片段必须逐字存在；字节列由 shape×dtype 推出并与
`tools/gen_ub_l1_budget.py` 的 workspace 总量对账（4353.69 MB 一致，`tests/test_stage_lifecycle.py`
把它固定下来）。

`[1,8192,96,128]`、C=64 下的 GM 流量（每调用）：

```text
总量           写 5042.65 MB   读 4636.85 MB   合计 9679.50 MB
host           808.50 / 808.50      （输入打包读入）
pre_gram      2425.36 / 2220.88
solve_wide     352.32 /  150.99
solve_assemble  25.17 /   25.17
solve_cube     402.65 /  402.65
k2            1028.65 / 1028.65
```

读法：**pre_gram 一家的 GM 流量（4.6 GB）就占全流水的一半**，因为它的 Gram 产物要在
AIC 与 AIV 之间走两趟 GM（AIC 落 raw fp32 → AIV 取回、mask/scale 后再落）。

### 3. 账本抓到的候选：三处 debug-only 写

把每行的 consumer 追一遍，发现生产路径上有三个缓冲区**没有 kernel 读**，只有
`return_intermediates` 的 debug dict 读：

| 写 | 字节/调用 | 谁写 | 为什么没人读 |
|---|---:|---|---|
| `Aqk32` masked 回写 | 201.33 MB | pre_gram AIV `post_gram`（select+scale 后 MTE3） | 生产只消费 `Aqk16`；`Aqk32` 只在 debug dict |
| `A32` | 201.33 MB | solve wide AIV（fp32 A_inv） | 生产只消费 `A16`；`A32` 只在 debug dict |
| `BetaOut` | 3.15 MB | pre_gram AIV | 同上 |

另有 `Aqk32` **raw 的一次复读**（201.33 MB）：AIV 的 band 循环要把 raw Gram 取回来做
mask/scale，这次读是结构上需要的（除非 mask/scale 搬到 Cube），先记着不当作候选。

按仓库自己的两个实测速率折算：

```text
0.095 ms / 201 MB store（R3 的 K2 v_new 守卫）  → 三处 0.191 ms，含复读 0.286 ms
0.2 ms / GB（§11.12 的 L2 流量价）              → 三处 0.081 ms，含复读 0.121 ms
```

两个数都远小于 §11.27 的 1.1 ms 上界，但**这是当前唯一"改动小、可位一致、有既有先例"的
候选**（K2 的 row-major `v_new` 就是同一个 pattern，§11.25 已落地）。判决：**允许进入
Level 1，但必须先做同进程交错 A/B**——两个速率差 2.4 倍，说明这 400 MB 里有多少是真暴露的
写带宽、多少已经被 MTE3 排水吃掉，只有量了才知道。

### 4. 设备健康：这次有 4 个 pin 的失败是环境，不是回归

本轮开始时 `npu-smi` 上 device 0 是 `Alarm`（`extra-info/data-dump/0/` 有 03:11 的
exception dump），在它上面跑 `tests/test_workspace_pool_numerics.py` 得到
`Vector core execution timed out` / `aclrtMemcpyAsync workspace input failed: 507034`，
4 个 pin 记在 `lastfailed` 里；换到健康的 device 1（`ASCEND_RT_VISIBLE_DEVICES=1`）同代码
重跑，`..` 全过。**规则：跑任何 device 侧 pin 之前先看 `npu-smi` 的 Health 列；Alarm 设备
上的失败不进结论。**

### 5. 这一轮之后还剩什么

| 方向 | 上界/代价 | 状态 |
|---|---:|---|
| S1 workspace 池化 | 0.30 ms，且埋在 ≥5 ms 的 host slack 里 | **冻结** |
| debug-only 三处写（Aqk32/A32/BetaOut） | 0.08～0.19 ms（不含复读） | **候选**（Level 1，需交错 A/B） |
| `Aqk32` raw 复读（mask/scale 搬 Cube） | +0.04～0.10 ms | 待定（要动 pre_gram 的 AIV band 循环） |
| Level 2 `pre_gram->solve` window 流式 | 省 L/Rk/Rv 的一趟 GM = 0.6 GB ≈ 0.12 ms（L2 价）**（§11.46 复价：跨 launch 读实测 1.43 ms/GB ⇒ ≈0.74 ms，6 倍）** | 未开工（先交 slot/credit 协议） |
| Level 3 `solve->K2` window 流式 | 省 W/U 的一趟 = 0.4 GB ≈ 0.08 ms**（§11.46 复价 ⇒ ≈0.50 ms，但吃 K2 的延迟墙与 stage 配平墙）** | 未开工 |
| **（§11.48 判决）** | 两级一起 **CLOSED**：Level 2 的唯一可行点净 **+0.22 ms（变慢）**，Level 3 的消费者装不上窗口且 credit 会死锁 | 不要再按字节账重启 |
| super-kernel / persistent scheduler | §11.27 判决：整条路线冻结 | **冻结** |

Level 2/3 的准入条件（slot 字节账、credit/free 协议、同步审计 diff、精度 gate、交错 A/B）
写进了设计书第 3 节；按 L2 价，这两级加起来也只值 0.2 ms 量级，因此**顺序是把 Level 1 的
三处写先量掉**，再决定要不要为 0.2 ms 重排跨 stage 交接。

**（2026-09-28 补充：这个排序所依据的"0.2 ms/GB"已被 §11.46 直接量翻——跨 launch 的读字节
在 cube 里的边际价格是 1.427 ms/GB，是 L2 价的 7.1 倍。Level 2 单级就有 ~0.74 ms。）**

## 11.29. 全流水生命周期重构（二）：三处 debug-only store 落地（-0.164 ms），以及一次空指针的教训（2026-09-23）

§11.28 的账本把它们排在最前面：一次调用里有三个缓冲区**没有设备侧读者**——`Aqk32` 的
masked fp32 回写（pre_gram AIV）、`A32`（wide solve 的 fp32 A_inv）、`BetaOut`——它们只服务
`return_intermediates` 的 debug 视图。当时按仓库两个实测速率折算成 0.081～0.191 ms，判决是
"候选，先做同进程交错 A/B"。这一节是那次 A/B 与落地。

### 1. 一次失败：不能用空指针关掉 `Aqk32` 的回写

第一版实现沿用 K2 `v_new` 的写法——生产传空指针、kernel 判 `pAqk32 != nullptr`。结果是设备
直接报错（`npuSynchronizeDevice ... SUSPECT REMOTE ERROR, error code 507057`）。原因很直接：
`Aqk32` 那块 GM **同时是 pre_gram AIV 的输入**——`post_gram` 的 band 循环要把 Cube 落下的
raw Gram 读回来做 mask/scale/round，只有"masked 回写"是没人读的。指针一为空，读的那条
`DataCopy` 就去访问地址 0。

改法：两个 kernel 各加一个 `int32_t debugStores` 尾参数，`api.py` 传
`1 if keep_debug else 0`；`keep_debug = return_intermediates or KDA_DEBUG_STORES == "1"`，
生产默认 0。**为什么是 flag 不是空指针**：一块 buffer 只要还有人读，它的生命周期就不能靠
指针是否为空来表达。`A32`/`BetaOut` 本来可以安全地用空指针（kernel 里只有写），但三个 store
用同一个机制更好审计。

### 2. A/B（`tools/probe_dead_store.py`）

同一进程、同一份编译产物，两臂只差 `KDA_DEBUG_STORES`（0/1），每轮跑两臂、取 `do_bench`
中位（warmup 100 ms / rep 800 ms），三轮：

| 轮次 | stock (10.7x) | guard | 备注 |
|---|---:|---:|---|
| 1 | 10.765 (p20 10.743 / p80 10.809) | 10.600 (p20 10.583 / p80 10.648) | |
| 2 | 10.754 (p20 10.735 / p80 10.805) | 10.603 (p20 10.587 / p80 10.641) | |
| 3 | 10.766 (p20 10.741 / p80 10.801) | 10.601 (p20 10.586 / p80 10.638) | |
| **中位** | **10.765** | **10.601** | **−0.164 ms（−1.5%）** |

两臂的臂内 p20/p80 只差 0.05 ms，三轮间差 0.012 ms（远小于 0.164 ms 的信号），并且
**输出与 final_state 逐位一致**（`torch.equal`）——跳过一块没人读的 store 不改变任何算术。
实测 −0.164 ms 落在两个估计值（0.081 / 0.191）之间：说明这 405.8 MB 里大约三分之二确实是
暴露的写带宽，其余被 MTE3 排水吃掉了。

### 3. 落地与门禁

- `kernels/v1/k1_pre_gram_mix.cpp`、`kernels/v1/k1_solve_wu_wide.cpp`：新增 `debugStores`
  尾参数，三处 store 受它控制（`Aqk32` 的读、`L` 的回写、`Aqk16` 的 Cast 全部不动）；
- `python/kda_ascendc_v1/api.py`：`keep_debug` 决定该参数，默认 0；`KDA_DEBUG_STORES=1`
  是回到旧行为的开关；`return_intermediates=True` 时三块照写（debug 视图不放未初始化内存）；
- `tests/test_dead_store.py`：三件事——flag 真的传到了两个 kernel（host 侧抓 launch 参数）、
  两臂逐位一致、`return_intermediates` 仍然填满三块 tile；
- 写这个测试时抓到一个**与本次改动无关的既有缺口**：`A32` 的 debug 视图只被写了一半。
  wide kernel 写对角线子块、并把严格上三角清零，但**左下角的耦合块 X21 是 assemble kernel
  算进 `Pmid` 的，没有任何 kernel 把它写回 A32**——那块是未初始化的设备内存，同一次调用的
  两次运行会给出不同的值（实测第一带 −2.68e-17 vs 0.0）。测试因此只比较"真被写过"的区域，
  并把这条记录在这里：`return_intermediates` 的 `A32` 视图在下三角的耦合块上是垃圾值，
  要修就得让 assemble 顺手写一份 fp32（+201 MB store，与本轮的目标相反），所以先记不修；
- 账本随改动更新：生产 GM 写 **5042.65 → 4636.85 MB**，总流量 **9679.50 → 9273.70 MB**
  （每调用、C=64），三行标为 `skipped`，`tests/test_stage_lifecycle.py` 断言它们不计入生产
  流量——这正是 §11.28 给账本定的规矩："落了守卫就要同一次改动里教会账本"。

**本轮的门禁**：`tools/run_chunk_matrix.sh`（C=16/32/64 三条腿）全绿；benchmark 的 golden
没有重录——本机是 `Ascend910B3` 而 golden 记在 `Ascend910_9382`（§11.26 的口径规则：跨设备
只比同一格式内的比值），跨设备的 gate 读数不属于本轮证据，A/B 才是。

落地时的一个教训，值得单独写下来：`keepA32` 的声明最初被放在 `#if KDA_SOLVE_WIDE_SUBB > 1`
里面，而它的第二个使用点在**单层路径**上——C=16/32 恰好只编译那条。于是 **C=64 的 A/B 全绿、
C=16 的 RTC 编译直接失败**（`aclrtcCompileProg failed`），是 `tools/run_chunk_matrix.sh` 的
C=16 腿抓到的。这和 §11.24 的 C=32 bug 是同一类：一个只在别的 build 里存在的代码路径，
必须靠三个 build 的矩阵去覆盖，不能靠"生产形状量过了"来推断。

### 4. 还剩什么

| 方向 | 上界/代价 | 状态 |
|---|---:|---|
| 三处 debug-only store | 实测 −0.164 ms | **已落地** |
| `Aqk32` raw 复读（201 MB，需要把 mask/scale 搬到 Cube） | +0.04～0.10 ms | 待定（要动 pre_gram 的 AIV band 循环） |
| Level 2 `pre_gram->solve` window 流式（省 L/Rk/Rv 一趟 GM） | ≈0.12 ms（L2 价）**（§11.46 复价 ≈0.74 ms）** | 需先交 slot/credit 协议 |
| Level 3 `solve->K2` window 流式（省 W/U 一趟） | ≈0.08 ms**（§11.46 复价 ≈0.50 ms）** | 同上 |
| **（§11.48 判决）** | 两级 **CLOSED**：knee（W ≤ 175）与一波字节（W ≥ 384）不相容；K2 `nblk == aic_cores = 24` ⇒ credit 自旋必死锁 | — |
| S1 workspace 池化 / super-kernel | §11.28 / §11.27 判决 | **冻结** |

下一档的账要诚实：Level 2/3 加起来约 0.2 ms，而它们要动的是跨 launch 的交接协议与同步审计，
成本远高于本轮这个 3 文件的守卫。**先把 Level 1 剩下的一条（`Aqk32` raw 复读，同一套账本
已经标好位置）量掉**，再决定要不要为 0.2 ms 重排 pre_gram->solve 的交接。

**（2026-09-28 补充："加起来约 0.2 ms"这个数是按 §11.12 的 L2 价算的，而 Level 2/3 省的恰恰
是**跨 launch** 的一趟 GM。§11.46 把这条边直接量了：1.427 ms/GB。收益侧应从 0.2 改成
0.74（Level 2）+ 0.50（Level 3），准入成本没变，排序应当重排。）**

## 11.30. 全流水生命周期重构（三）：state/output 分离的判决实验——分割合法、逐位一致，但总价 +1.70 ms（+16.1%）（2026-09-23）

用户在设计书（`docs/PREFILL_LIFECYCLE_REFACTOR_20260922.md`）之外给的三条路线里，第一条是
"把 output 从串行状态链里拆出去"，并指定了判决口径：**保留现有 K1、只换 K2，比较现有 K2
vs state-only + parallel-output，并且必须计入两个 kernel、快照、布局转换和输出的总成本**。
"单独 state kernel 更快不算成功。"这一节就是那次判决。

### 1. 落地

- `kernels/v1/k2_state_loop.cpp`：`k2_persistent_loop` 去掉全部 output 侧——不读 `Qg`、不做
  `Qg@S^T`、不读 `Aqk`、不做 `d3 = Aqk@v_new`，stage 4 只剩状态递推。bf16 状态发布从"一个
  被反复覆盖的 S16 槽"改成**按 chunk 索引的 `Hsnap[task, chunk]`**：同样的字节、不同的地址，
  所以**序列链为快照付出的写侧成本是零**；设计书里那 384 MiB 的快照价只落在读侧。
- `kernels/v1/k2_out_parallel.cpp`：全并行输出核。同一套 block/flag 骨架去掉 `FL_V`（AIV 不再
  生产 Cube 的operand），每 tile 两个 Mmad（`Qg@H^T`、`Aqk@Z`），并且**保留原 kernel 的每一个
  舍入位置**：Cube 把 d2/d3 舍成 bf16，AIV 用 fp32 累加 `out = d2*scale + d3` 后只舍一次。
- `python/kda_ascendc_v1/api.py`：`k2_mode="split_state_out"`（`SPLIT_STATE_OUT`），只走
  experimental 入口，公共 API 仍然拒绝；两个 launch 都在调用者 stream 上（见第 4 节）。
- 准入数（`tools/gen_ub_l1_budget.py` 现在把两个 kernel 都算出来）：

| kernel | UB | L1 | L0C | L0A / L0B | 结论 |
|---|---:|---:|---:|---:|---|
| `kda_k2_state_loop` | 184.5 KB | 80.0 KB | 96.0 KB | 16 / 32 KB | 全部在 cap 内 |
| `kda_k2_out_parallel` | 56.0 KB | 72.0 KB | 64.0 KB | 24 / 48 KB | 全部在 cap 内 |

- `tools/probe_state_out_split.py`：同进程交错 A/B（同一份编译产物、每轮两臂交替、`do_bench`
  中位、600 ms rep），外加一份**由几何推出的 GM 账**（两个 kernel 各自读写了什么）。

### 2. 判决（C=64，`[1,8192,96,128]`，三轮交错）

| 轮次 | fused | split | 备注 |
|---|---:|---:|---|
| 1 | 10.599 (p20 10.579 / p80 10.646) | 12.299 (p20 12.283 / p80 12.316) | |
| 2 | 10.600 (p20 10.574 / p80 10.623) | 12.306 (p20 12.283 / p80 12.344) | |
| 3 | 10.586 (p20 10.579 / p80 10.624) | 12.301 (p20 12.279 / p80 12.321) | |
| **中位** | **10.599** | **12.301** | **+1.701 ms（+16.1%）** |

- **两臂逐位一致**（`torch.equal`：out 与 final_state 都过）。这一步让"分割"的合法性从代数
  变成实现事实：两个新 kernel 复现了 shipped kernel 的每一个 bit，包括 d2/d3 的 bf16 舍入与
  `out = d2*scale + d3` 的 fp32 累加顺序。
- 同一次运行的 profile（每个 mark 都带 sync，绝对值偏高，只用来看内部比例）：
  `k2_ms` fused 4.078 vs split 5.928；拆开是 **state 3.446 + out 2.373**。
- GM 账（几何，无实测）：fused 5140.1 MB → split 5542.8 MB，**+402.7 MB**（正好是快照的读侧：
  `[tasks, nt, BV, D]` bf16 = 402.65 MB；写侧与 fused 的 S16 同量，不重复计账）。
- 小形状（`[1,512,4,128]`）同向：fused 1.068 vs split 1.105（+3.5%，受 launch/固定开销支配）。

### 3. 为什么输

1. **状态链只减重 15%**：删掉的是一整个 `Qg@S`（2 Mmad/head-chunk）、`Aqk@v_new`
   （2 Mmad/head-chunk）、`Aqk` 的 100 MB 读、`Qg` 的 201 MB 读、D2/D3 的 786 MB 往返和
   192 MB 的 out 写出——合起来只值 4.078 → 3.446 ms。也就是说 K2 的时间不在这些 Mmad 与
   这些 store 上，而在分离根本不碰的地方：stage 2 的 16 次转置与 Vt 打包、D4 的 fp32
   805 MB 写 + 805 MB 读、以及 per-chunk 的描述符数量（kernel 头注释早已记过：这个 loop 是
   descriptor/work-bound，不是 flag-bound）。
2. **并行输出核不便宜**：2.373 ms 里是 38.6 GFLOP 加约 1.85 GB 的 GM 往返，等于 ~16 TFLOPS
   有效吞吐。它是"更连续、更规则"的 Cube 计算没错，但本机没有便宜到能让这条路线翻盘。
3. **快照的读侧是实价**：fused 的单槽 S16 只有 3 MB、常驻 L2、几乎免费；按 chunk 物化后，
   输出核要把 384 MiB 从 HBM 再读一遍（+0.33 ms HBM 价、+0.08 ms L2 价），而它换来的
   只是"输出可以晚算"这一条自由度。

### 4. 边界与残留

- 唯一可能翻盘的形态是**输出核与状态核在设备侧重叠**（按 chunk 的 ready 计数、跨 launch
  的 hand-off，也就是 Level 4 persistent scheduler）。本次两段在同一 stream 上串行，这是
  设计书明确后置的形态，不在这条候选的范围内——但它是这条路线唯一的翻盘点，而不是
  "换更小的 tile"。
- 候选留在树里，并由 `tests/test_state_out_split.py` 钉住三件事：与 fused **逐位一致**、
  确实是两次 launch（`kda_k2_state_loop` + `kda_k2_out_parallel`，且没有 `persistent_loop`）、
  公共入口继续拒绝它。C=16/32/64 三条腿都跑（和 §11.29 同一个理由：只在别的 build 里存在的
  路径必须靠矩阵覆盖）。
- 这条判决同时给后面所有"用更多 GEMM 换更短依赖链"的路线**定了一个价**：本机的稠密并行
  Cube kernel 实测大约值 16 TFLOPS 有效（含 GM 往返），不是峰值。§11.31 用它算路线 2 的账。

## 11.31. 稠密仿射递推（路线 2）的精度门：`diag(d)` 不能进 bf16 转移矩阵（2026-09-23）

路线 2 把递推改写成

```text
E = diag(d) - Kg^T W     F = Kg^T U     P = Q - A W     R = A U
Snext = E S + F          O = P S + R
```

它与 shipped 的 `Z = U - W S`、`Snext = diag(d) S + Kg^T Z` 代数等价，但**不是 bf16 等价**：
shipped 用 fp32 的 decay 乘 fp32 的状态，仿射路径的整个转移（包括 decay 本身）必须变成 Cube 的
bf16 operand。设计书要求"先过长链精度再测完整成本"，这一节就是那个门。

### 1. 工具与口径（`tools/probe_affine_precision.py`）

一次 fp32 chunk 链做参考，然后让四种 rounding discipline 消费**同一组 fp32 中间量**：

| 名称 | 含义 |
|---|---|
| `fp32` | 无中间舍入（参考本身） |
| `shipped` | 现实现：bf16 W/U/Aqk/Qg/kg、bf16 S16 快照、bf16 d1 与 Z、**精确 fp32 decay**、fp32 累加状态 |
| `affine` | E/P 舍成 bf16（Cube operand），F/R 保持 fp32 |
| `affine_dec` | 同上，但 decay 留在矩阵外：`Snext = diag(d) S + bf(E_off) S + F` |
| `affine_fp32` | E 不舍入（上界，用来区分"是 bf16 的错"还是"是形态的错"） |

gate regime：C=64、T=8192、H=8、B=1，三种输入（`long` 长记忆 / `mid` 中等衰减 / `init` 非零初值），
工具会打印每个 regime 的实测 per-chunk decay `2^Σgate`。

### 2. 结果（state / out 相对误差，vs fp32 参考）

| regime | shipped | affine | affine_dec | affine_fp32 |
|---|---|---|---|---|
| long | 4.41e-3 / 6.14e-3 | **5.83e-3** / 6.32e-3 | **3.32e-3** / 5.98e-3 | 3.08e-3 / 5.26e-3 |
| mid | 4.26e-3 / 8.13e-3 | **7.32e-3** / 6.23e-3 | **3.49e-3** / 5.36e-3 | 3.28e-3 / 5.24e-3 |
| init | 4.79e-3 / 5.65e-3 | **6.40e-3** / 5.49e-3 | **3.69e-3** / 4.82e-3 | 3.20e-3 / 4.72e-3 |

判决：

1. **把 `diag(d)` 折进 bf16 的 E 不合格**：状态误差比 shipped 高 30～70%（`affine` 与 shipped
   的差值本身就有 5.2～7.7e-3 相对量级，和中/长记忆 regime 下 shipped 的整份误差同阶）。
   bf16 只有 8 位有效位，对角线在 1 附近的实现在每 chunk 引入 ~2^-9 的系统性偏差，长链上
   直接变成状态漂移。
2. **decay 留在矩阵外合格**：`affine_dec` 的状态误差 3.3～3.7e-3，**比 shipped 低 25～30%**，
   输出误差也略低——因为仿射路径反而省掉了 shipped 交的两次舍入（d1 与 Z）。也就是说
   路线 2 的数值可行性存在，但形态被钉死成"fp32 状态 + fp32 decay 留在 AIV，只有
   `E_off = -Kg^T W` 进 Cube 的 bf16 operand"。
3. 没有出现长链爆炸：128 个 chunk、弱衰减、非零初值都稳在同一量级。
4. **口径教训**：仓库默认输入（`g = logsigmoid(randn)`、`A_log = randn(H)`）的 per-chunk
   decay ≈ 0（状态在一个 chunk 内就忘光），在这种输入下任何 formulation 都看不出差别——
   一个看不出差别的实验会伪装成"通过了"。工具因此显式构造 long-memory regime 并打印
   `2^Σgate`。
5. **方法教训**：这版模拟器的前两稿有 bug——状态 `(V,K)/(K,V)` 装反、`P` 漏了 scale——
   两次都是靠"同时和 shipped 比一份副表"抓到的（只看 vs fp32 的话，错误会被读成
   "仿射路径误差 100%，判决不通过"）。数值研究必须自带同形态的影子对照。

### 3. 代价侧（判决的另一半）

设计书口径：目标形状主要 GEMM 从 ~90 GFLOP 增到 ~155 GFLOP（+59 GFLOP，不含公共 K1），
而且 E/F 的构建在**每个 chunk 的串行链上**。按 §11.30 实测的价位（38.6 GFLOP 的稠密并行
kernel 单独跑 2.373 ms ⇒ ~16 TFLOPS 有效，其中约 1.5 ms 是 GM 往返），这 +59 GFLOP 值
1.5～3.7 ms，而 K2 的全部只有 4.08 ms。

结论：**路线 2 不进入实现**——它的数值形态被精度门限死（decay 必须在 AIV 的 fp32 里，
所以跨核交接一点不少），它的代价按本机实测价位收不回来。除非出现新的硬件事实（例如
`L0C→A1→L0B` 反馈能稳定省掉一趟 GM——仓库记录过该路径可用，但独立探针没有稳定加速），
这条路线没有第二次机会。

## 11.32. 三条路线的位置与下一步（2026-09-23）

| 路线 | 状态 | 证据 |
|---|---|---|
| 1. state/output 分离 | **判决：不采用**（+1.70 ms / +16.1%，逐位一致） | §11.30，唯一翻盘点是 Level 4 的设备侧重叠 |
| 2. 稠密仿射递推 | **判决：不采用**（精度门只在 decay 留 AIV 时通过；+59 GFLOP 按实测价位收不回） | §11.31 |
| 3. 融合式分块 solve | **未测**，入场条件见下 | 设计书 §3 + 本节的定价 |
| 4. C128 / tile 解耦 | **未测**，是几何候选而非承诺 | 设计书 §4 |
| 5. segment scan | 后置（小 B/H、超长 T 的专用分支） | 设计书 §4 末 |

路线 3 要动 K1，它的对手不是"空"，而是已上线的两级 solve：隔离口径下 AIV 半 2.40 ms、
AIC 半 2.59 ms（设计书 §1）。blocked TRSM 变体能省的只有"应用"那半的一部分：按 C=64、
leaf 16 的口径估，Cube 侧 MAC 从 `64x64x256`（=1.05 M/chunk）降到两次 leaf 应用加一次
耦合更新（≈0.78 M/chunk），外加上省掉 a16 的 100 MB 写 + 100 MB 读，即 ~25% 的 Cube 工作
与 ~0.3 ms 的带宽——而 leaf inverse 的批量计算、宽 RHS 的列切片、以及本来就在 AIV 上的
per-chunk 串行一样都没少。设计书转述的旧 TRSM 原型是 11.45 ms（本轮仓库里没有对应源码，
按设计判断引用），它否定的是"TRSM 形状不对"的实现，不是这个方向的全部；但在拿到
"leaf inverse 批量 + 块更新走 Cube + 中间 RHS 不落 GM"的一版实现之前，**不建议动 K1**。

下一步的顺序因此是：路线 3 先做**一份不落 GM 的 leaf-inverse + 块更新设计**（含 slot/credit
与 UB 预算），再决定是否写 kernel；路线 4 先试不动协议几何的变体（C=96，或"逻辑 chunk 64 +
值 tile 128"）；路线 5 保持在专用分支的位置。

## 11.33. 三条路线同轮执行（2026-09-23 晚）：C128 装不下且不回本、宽 RHS solve 的 AIV 地板已超整段预算、checkpoint/replay 被本轮实测的 GM 价位判死

本轮顺序按设计书 §4：分层 C128 → 融合式宽 RHS solve →（组合）→ checkpoint/replay。
三条都跑到了判决，**没有一条进入生产**，`python/kda_ascendc_v1/api.py` 的生产路径一字未动。

### 1. 分层 C128：可行性算术 + "上一次同实验"的实测

**(a) 设备事实。** `KDA_CHUNK=128` 全链路 RTC 编译通过（`_defines`：CHUNK 128 /
SOLVE_WIDE_NCHUNK 8 / SUBB 2 / ASM_NCHUNK 4 / WU_NCHUNK 2 / MAXH 2），但第一个 launch 就把
设备打挂（`tools/probe_chunk128_k1.py`，日志 `/tmp/c128_run.log`）：

```text
FAULT kda_pre_gram_mix  blocks=2
       -> 507015 aicore exception, core id 33
          "The GM address accessed by scalar exceeds 48 bits"
```

`/tmp/bisect_c128.py` 在每次 launch 后加一次 `torch.npu.synchronize()`，把它定位到
`kda_pre_gram_mix` **自己**（不是 solve、不是 K2）；同一个 binary 在 C=64 一切正常。所以
"当前实现装不下 C128" 是 kernel 的 UB/L0 算术问题，不是数学问题——与设计书的判断一致。

**(b) 算术（可复算）。** `tools/gen_ub_l1_budget.py` 新增 `pre_gram_ub()`：直接从
`kernels/v1/k1_pre_gram_mix.cpp` 的 `InitBuffer` 表达式求值（AIV 半边一份账、Cube 半边
A1/B1/CO1 一份账，两者在同一个文件但不在同一块存储里）：

| CHUNK | AIV half (UB) | Cube half (L1 queues) | L0: A2 / B2 / CO1 |
|---:|---:|---:|---|
| 16 | 119.7 / 192 KB（−72.3） | 42.0 / 512 KB | 16 / 8 / 2 KB |
| 32 | 141.5 / 192 KB（−50.5） | 88.0 / 512 KB | 32 / 16 / 8 KB |
| 64 | **185.3 / 192 KB（−6.7）** | 176.0 / 512 KB | 64 / 48 / 32 KB |
| 96 | 229.7 / 192 KB（**+37.7**） | 280.0 / 512 KB | 96 / 64 / 72 KB（L0B 已在 64/64） |
| 128 | **274.6 / 192 KB（+82.6）** | 400.0 / 512 KB | **128 / 80 / 128 KB** |

C=64 的 185.3 KB / 余 6.7 KB 与 kernel 注释里实测的 "under 8 KB of UB headroom" 对上
（模型不计 TPipe 的对齐/padding），说明它没有系统性偏差。C=128 同时超三项硬件预算：UB
+82.6 KB、L0A 128/64 KB、L0B 80/64 KB（L0C 正好占满 128/128）。超预算的四块就是分层方案
要动的地方：`bT0`（整 chunk gate cumsum，64 KB）与 `qgin/qgmk/qgout`（16 行 × M 列的 band
staging，各 32 KB）。

**(c) 判决规则触发（"如果 C128 的 K1 已经没有收益，就不应该继续改 K2"）。** "chunk 数减半、
M 翻倍"这个实验在上一步已经量过两次：

| 口径（`[1,8192,96,128]`，C=32 → C=64） | K1 | K2 | 证据 |
|---|---:|---:|---|
| 文档同版本交错 A/B（§11.28） | pre 3.189 → 3.445，solve 2.294 → 2.548 = **+0.51 ms** | 3.405 → 2.462 = **−0.94 ms** | 同进程交错 |
| 本轮同机同日 K1-only（do_bench 中位 of 3） | 6.326 → 6.642 ms = **+0.316 ms** | — | `tools/probe_chunk128_k1.py` |

（本轮三条腿：C=16 6.589 / C=32 6.326 / C=64 6.642 ms，每 1000 token 0.804 / 0.772 / 0.811 ms。）
也就是说，**上一步里 K1 已经开始倒亏**：增长的机制是算术的（pre_gram 的 Gram/post_gram
元素数、solve 的行递归 lane 数都随 M 上升），而 C=64 时每 chunk 的固定成本已经摊薄。因此
C=64 → C=128 不可能让 K1 变好，只剩 K2 还能赚——上限就是上一步实测的 −0.94 ms，而 K2 的
每步成本本身也随 M 上升。**冻结 C128，不为它动 K2。**

**(d) 将来重开的入口条件**（写清免得重新论证）：把 `post_gram` 的 band staging 从
`[16, M]` 切成两半 `[16, 64]`（或降成单槽 ring，省 ~72 KB）；把 `bT0` 的整 chunk gate
cumsum 改成带 carry 的 32 行 band（省 48 KB；**加法顺序变了，要重跑数值 gate，不是逐位
一致**）；把 L0A/L0B 的操作数切片各减半。三条都做完只是"装得下"，回本仍要按 (c) 的账算。

### 2. 融合式宽 RHS solve：AIV 半边的**地板**已经超过整段预算

`tools/probe_rhs_substitution.py` + `kernels/v1/k1_solve_rhs_probe.cpp`（计时探针，非生产
kernel；只写 32 B 防 DCE，没有 gather / cast / store）。同进程、12288 个 chunk-instance
（生产 chunk 数）、grid 1536、MIN of 5：

| 臂 | 指令/chunk | repeats | ms | ns/chunk |
|---|---:|---:|---:|---:|
| null | 0 | — | 0.081 | 6.6 |
| shipped shape（M=32、32 lane、8 实例） | 124 | 8 | **1.169** | 95.2 |
| fused 2×32（256 lane RHS、4 实例） | 496 | 8 | **5.854** | 476.4 |
| fused 4×16（256 lane RHS、8 实例） | 240 | 8 | **2.941** | 239.4 |

同一进程重放生产的 solve launch：**AIV 2.123 / AIC（assemble+cube）2.587 / 重叠 2.644 ms**。

判决：停止规则是"完整融合 solve ≥ ~2.5 ms 即停"。两个融合形态的 AIV 地板——**不含** RHS
gather、**不含** W/U store、2×32 还**不含**耦合那步 Cube——已经是 2.941（1.11×）与 5.854
（2.21×）ms。⇒ **路线停止，不写 Cube 半边。**

机制（为什么不是实现问题）：行递归的指令数由分块决定（每个对角块 `Σ_i i`），而**每条指令
的宽度**由另一个操作数决定。现实现递归在 M×M 的 inverse 上（32 lane × 8 个 chunk 实例，
UB 32 KB），融合形态递归在 `[M, 256]` 的 RHS 上（256 lane × 4 个实例，同样 128 KB UB），
于是每 chunk 的指令数 ×4；4×16 把指令数砍半（240）但仍然在整段预算之上。这正是 §11.9
"solve 的 2.53 ms 是真算力" 的另一面：**inverse 是小操作数、宽 repeat；RHS 是大操作数、
窄 repeat**，而 UB 只够一种。

副产品（对后续有用）：现实现 AIV 半边的 2.123 ms 里，递归地板只有 1.169 ms（55%），剩下
~0.95 ms 是 L gather、bf16 往返 cast 与 A16/Xb/Lneg store。要再动 solve，**先动这 0.95 ms**
（例如 A16 store 只写 Cube 真正读的对角块），而不是重排递归。

### 3. checkpoint / replay：用本轮实测的 GM 价位直接判死（**定价否证**，不是实测）

§11.30 的分割实验给出了"额外 GM 流量"的实测价格：+402.7 MB（bf16 快照读侧）= +1.70 ms
⇒ **4.2 ns/byte**（≈236 GB/s 有效）。interval = 4 的账：

| 项 | 量 | 价 |
|---|---:|---:|
| 省：快照流量 402.7 → ~100 MB | −300 MB | −1.27 ms |
| 加：output 重放要再读 W/U/Qg/Kg/Aqk | +908 MB | +3.8 ms |
| 加：重算 Z、A@Z、Q@H | ~39 GFLOP | +2.4 ms（按 §11.30 的 ~16 TFLOPS 有效价） |

⇒ 净 **+5 ms 量级**，且 final state 仍然只能串行。**不做 interval sweep**。重开条件只有
一个：诊断显示"快照 workspace 或它自己的流量"是瓶颈——而本轮的测量正好是反证：把快照流量
开到最大的 split 变体（§11.30）已经更慢。

### 4. 三条路线之后的账面

| 路线 | 本轮状态 | 下一步 |
|---|---|---|
| 分层 C128 | **不采用**：装不下（UB +82.6 KB、L0A/L0B 各超），且上一步同实验里 K1 已倒亏 +0.51 ms | 入口条件见 1.(d)；重开前先按 1.(c) 的账回本 |
| 融合式宽 RHS solve | **不采用**：AIV 地板 2.94 / 5.85 ms ≥ 整段 2.64 ms | 若要再动 solve，先削 AIV 的 0.95 ms 非递归开销（2 的副产品） |
| checkpoint / replay | **不采用（定价）**：净 +5 ms 量级 | 只在诊断显示 workspace/流量是瓶颈时重开 |
| 组合（C128 + 新 solve） | 随前两条一起冻结 | — |

生产路径未改；本轮新增的都是探针与账本工具（`tools/probe_rhs_substitution.py`、
`tools/probe_chunk128_k1.py`、`tools/gen_ub_l1_budget.py` 的 `pre_gram_ub()`、
`kernels/v1/k1_solve_rhs_probe.cpp`）。

## 11.34. solve 的 0.95 ms 判决：A16 store 消融——那 0.95 ms 是被 AIC 遮住的松弛量，不是 stage 时间（2026-09-23 深夜）

§11.33 的副产品把下一步钉成"若要再动 solve，先削 AIV 半边那 0.95 ms（递归地板 1.169 之外的部分）"，
本轮就按它执行：A16 store/blank 消融 → 再看 Xb/Lneg 能不能直供 assemble → 布局不动。
**结论是这条分支关闭**，而且理由是反的：删掉 parent tile 的写回不但不赚，端到端还稳定慢 ~0.2 ms——
因为 solve 的 stage 早就被 AIC 半边定住（2.535 ms ≈ ASM 0.928 + CUBE 1.586），AIV 的 2.144 ms 是**被
遮住的松弛量**，不是 stage 时间。

### 1. 消融怎么做：一个运行期 int32，不是重编译

`kernels/v1/k1_solve_wu_wide.cpp` 的签名在 C 与 debugStores 之间多了一个 `a16Mode`：

| 模式 | 写什么 | 用途 |
|---:|---|---|
| 0 | 生产：parent tile 的两个对角子块 + 严格上三角的 blank（Cube 当 0 读） | 默认，生产路径 |
| 1 | 只写对角子块，不写 blank | 消融 |
| 2 | parent tile 一个字节都不写 | 消融 |

`api.a16_mode()` 每次调用读一次 `KDA_SOLVE_A16_MODE`（不在 import 时冻结，探针才能在一个进程里翻臂），
缺省 0 就是生产；`tests/test_solve_a16_ablation.py` 钉住接线（生产恒为 0、所有 slice 带同一个模式、
debug flag 仍在最后一个 int）。**模式 1/2 是故意错的**：`kda_solve_wu_cube_kernel` 会读到部分写入的
A_inv，探针的 `--cold` 量的就是这件事——冷 A16 下 mode 2 与 mode 0 有 100663008/100663296 个元素不同
（max|d| 9.399e-03）；热跑时三臂逐位相同，只是缓存分配器把上一轮那 98 MB 原样还了回来。

### 2. 三臂（[1,8192,96,128]、C=64、同进程、`tools/probe_solve_a16_ablation.py`）

| 口径 | mode 0 生产 | mode 1 去 blank | mode 2 去整个 parent tile |
|---|---:|---:|---:|
| e2e（do_bench 中位 of 3） | 10.623 ms | 10.664（**+0.041**） | 10.847（**+0.224**） |
| 生产 schedule 重放（24 slice + per-slice event） | 2.535 | 2.533（−0.002） | 2.620（**+0.085**） |
| AIV（wide kernel） | 2.144 | 2.053（−0.091） | 1.862（−0.282） |
| ASM（assemble） | 0.928 | 0.928 | 0.929 |
| CUBE（solve） | 1.586 | 1.587 | 1.588 |
| AIC（ASM+CUBE 同一条 stream） | 2.614 | 2.616 | 2.615 |

e2e 的符号在三个独立进程里一致（正序 +0.023/+0.216、倒序 −0.004/+0.197、正序 +0.041/+0.224）：
blank 那一份无事，parent tile 那一份稳定慢 ~0.2 ms。

删掉的字节：blank 2 KB/chunk + 对角 4 KB/chunk = 6 KB/chunk ×12288 = **73.7 MB/call**；AIV 因此少
0.282 ms ⇒ **3.8 ns/B**，与 §11.30 用另一条路径量到的 4.2 ns/B 同量级：**store 没有变慢，它只是不落在
关键路径上。**

### 3. 机制：stage = max(AIV, AIC)，而 AIC > AIV

- 按生产 schedule 重放（每 slice：sa 上 wide → event → sb 上 assemble+cube）得 **2.535 ms**，
  与 api 注释里记的 24-slice 值 2.525 对上；
- 单独重放 AIC 半边（ASM 0.928 + CUBE 1.586 = 2.514）得 2.614 ms（差值是 launch 之间的空隙）；
- 也就是 per-slice 的双流重叠已经把 AIV 整个藏住，只多 0.02 ms：**stage 就是 AIC 的串行工作量**。

为什么删了反而更慢：AIV 提前做完，它的 L gather 和剩下的 store 就与 AIC 那一刀挤进同一段时间
（§11.27 已量到这台机器是吞吐饱和的），两组访存在内存系统上互相拖。证据是三个模式**单独**跑 AIC 时
一模一样（2.614/2.616/2.615），差异只能来自干涉；`overlapped`（无 event 的并发重放）与 `sliced`
（生产 schedule）两个口径同向变差。

### 4. 判决：分支关闭；步骤 2（Xb/Lneg 直供 assemble）也不上场

- Xb 只有 4 KB/chunk（50.3 MB/call），比 mode 2 删掉的那 6 KB 还小，而 mode 2 已经证明删更大的
  那份不赚；
- Xb 与 A16 的对角子块**同源逐位相同**（同一个 `ab[t*M]`，只有目的 pitch 不同），所以"能不能直读"
  是布局问题不是数值问题；
- 但让 assemble 从 A16 读对角块，是把一次连续的 2 KB `Nd2Nz` 换成 32 行 × 64 B 的跨步读——**往
  瓶颈那一侧加活**，方向反了；
- 设计书的准入"完整 solve stage 接近 2.53 ms 才继续"现在读作：stage 已经是 2.535，而唯一能在 AIV
  上省的时间被遮住 ⇒ **停**。布局（[row][chunk][lane]、Brcb + repeat-stride）本轮一字未动。

### 5. 本轮留下的新账：solve 第一次被拆成 AIV / ASM / CUBE

| 半边 | ms | 说明 |
|---|---:|---|
| AIV wide | 2.144 | 递归地板 1.169（§11.33），其余 ~0.98 是 gather/cast/store——**被遮住** |
| AIC assemble | 0.928 | 只做 X21 = X22·Lneg21·X11，两趟；GM 账面 12 KB/chunk = 151.0 MB ⇒ 163 GB/s，**不在带宽上**，是 wave/依赖受限（§11.35） |
| AIC cube solve | 1.586 | 80 KB/chunk = 1006.6 MB：A16 16 KB（**两个 pass 各读一遍**，其中一遍 100.7 MB）+ RHS 32 + W/U 32 ⇒ 635 GB/s（冷路径边际 839 GB/s，§11.35） |

**实测的否定**：想靠"把 assemble 的块做肥"降它的 wave 数也走不通——`KDA_ASM_NCHUNK = 6` 与 `8` 在
第一次 api 调用就把核挂住（AICore 100% 空转：两个设备各一次并发复现，外加一次单进程复现），与
kernel 注释里"L1 队列必须恰好 NC 深"是同一类约束。杀掉进程后设备恢复 0%。所以 assemble 的 NC 被
钉在 4。**（2026-09-24 补充：这条挂核只属于当时唯一的 shipped 队列形态；§11.37 的显式 buffer 下
6/8/12/16 都能跑且位一致，复测见 §11.41——但也没有收益，NC 仍是 4。）**

**剩下的两个入口（都在 AIC，不在 AIV）**：

1. **cube 每个 pass 重读 A16**：`qa` 只有 NC=2 个 L1 slot，两个 pass 各从 GM 读 8 KB/chunk，合计
   100.7 MB（它自己 1006.6 MB 的 10%）。让两个 pass 共用一次加载（L1 多 16 KB）⇒ §11.35 已把候选
   做出来量：**−0.070 ms**，不是投影的 −0.16 ms——第二次读是 L2 命中，只有它值得省。
2. **assemble 整块 0.928 ms**：字节账只有 151.0 MB（163 GB/s），削流量不会等比例回本；要动就得动
   结构——把 X21 的构造并进 cube kernel（上界 −0.93 ms，但那是 §11.9/§11.24 立起来那套东西的
   改写，需要自己的探针和 gate）。

生产路径未改（`KDA_SOLVE_A16_MODE` 缺省 0；本轮 C=16/32/64 的 chunk 矩阵全 PASS，bench gate 在本机
仍只差 §11.26 记录的跨设备绝对时间）。本轮新增：`tools/probe_solve_a16_ablation.py`、
`tests/test_solve_a16_ablation.py`、wide kernel 的 `a16Mode` 参数与 `api.a16_mode()`。

## 11.35. 探针：Cube solve 的两遍 A16 重读只值 0.070 ms（不是投影的 0.16），而 assemble 那 0.75 ms 才是 AIC 里没被解释的部分（2026-09-23 深夜）

§11.34 在 AIC 半边留下两个候选，第一个是 `kda_solve_wu_cube_kernel` 每个 pass 都从 GM 重读一遍
A16（8 KB/chunk/pass，共 100.7 MB，占它自己 1006.6 MB 的 10%）。它值不值那 0.16 ms 的投影，取决于
"这个 kernel 是不是带宽受限"——本轮不去推断，直接把候选做出来量。

`kernels/v1/k1_solve_wu_cube_a16_probe.cpp` 是 shipped kernel 的逐字转写（同样的 RHS 装载、两次
LoadData 交叉、Mmad、Fixpipe、L0/L0C 槽位与 InitBuffer 算术），三臂只差 A16 的装载路径；
`tools/probe_solve_cube_a16.py` 在同进程里交错跑三个臂，并重放生产的 cube launch。

| 臂 | A16 装载 | GM bytes/call | ms（MIN of 5，交错） | 边际 |
|---|---|---:|---:|---|
| mode 0 控制（shipped 结构） | 每 pass 各一次 | 1006.6 MB | 1.465 | — |
| mode 1 候选（A16 常驻 L1） | 每块一次 | 906.0 MB | **1.395** | −0.070 ms（后一遍 100.7 MB） |
| mode 2 地板（完全不读 A16） | 0 | 805.3 MB | 1.275 | −0.120 ms（冷的那一遍 100.7 MB） |
| 生产 cube（同进程重放 24 个 slice） | 每 pass 各一次 | 1006.6 MB | 1.571 | +0.107 vs 控制 |

判读：

- **候选值 0.070 ms**（1.465 → 1.395，交错 MIN of 5），不是投影里的 0.16 ms。机制在边际率上：
  第二遍读是 **L2 命中**（+100.7 MB 花 0.070 ms ⇒ 1439 GB/s 边际），第一遍是冷 HBM 读
  （+100.7 MB 花 0.120 ms ⇒ 839 GB/s 边际）。**只有 L2 那一遍值得省**，而候选省的正是它。
- **cube 自己已经贴着 HBM 速率**：冷路径边际 839 GB/s、绝对 687 GB/s；它还剩下 ~0.3 ms 的非 DMA
  时间（906 MB 按 839 GB/s 应是 1.08 ms，实测 1.395），那是另一类问题，不是这次要动的。
- 顺带量到 **slice 的价格**：同一份 1006.6 MB，生产的 24 个 launch 比单次 6144-block 启动贵
  **0.107 ms**（≈4.4 µs/launch）——这是 SOLVE_OVERLAP=24 那个旋钮的另一半账（§11.25 的 sweep 只
  看了重叠收益）。
- **更重要的对照**：assemble 是 151.0 MB 跑 0.928 ms ⇒ **163 GB/s**，按 cube 的冷路径边际率它只要
  0.18 ms；也就是说 AIC 半边里 **~0.75 ms 是 assemble 的结构开销**（两趟 + P 往还 + 每块 4 chunk），
  是这次 A16 那 0.070 ms 的十倍。§11.34 试过的"把块做肥"（KDA_ASM_NCHUNK = 6/8）会挂核，所以下一刀
  要么先给 assemble 做同款 DMA 结构探针（把它定住在小传输 / 两趟屏障 / wave 数上的哪一格），要么直接
  做把它并进 cube kernel 的重写。

**单位更正**：§11.34 与设计书 §4.7 的第一版把 KiB 计数当成了 MB——正确值是 cube 1006.6 MB / 635 GB/s、
assemble 151.0 MB / 163 GB/s、A16 重读 100.7 MB、assemble 每个 chunk 12 KB。比例、投影与判决不变
（0.16 ms 的投影本来就用同一套比例算出，实测值是 0.070 ms）。

本轮新增：`kernels/v1/k1_solve_wu_cube_a16_probe.cpp`、`tools/probe_solve_cube_a16.py`。
生产路径未改——**resident 形态还没有进生产 kernel**，它值 0.070 ms（cube 的 4.8%、solve stage 的
2.8%、e2e 的 0.7%），要落地得先跑 C=16/32/64 的矩阵与 bench gate。

## 11.36. 探针：assemble 的 0.75 ms 里，小传输自己就占 0.339（52%），两趟结构不花钱、P 往还只值 0.137（2026-09-23 深夜）

§11.35 把 assemble 定为 AIC 半边最大的未解释项：151.0 MB 跑 0.928 ms（profiler，带竞争）⇒ 163 GB/s，
按 cube 的冷路径边际率（839 GB/s）它只要 0.18 ms。这一轮不推断结构，把候选逐个做出来量：
`kernels/v1/k1_solve_assemble_probe.cpp` 是 shipped kernel 的逐字转写（同样的 6 次 ND2NZ、5 次
LoadData + 1 次 LoadDataWithTranspose、Mmad、Fixpipe、L0/L0C 槽位与 InitBuffer 算术、两趟结构），
五臂每次去掉一个候选；`tools/probe_solve_assemble.py` 在同进程里交错跑五臂（MIN of 5），并把生产的
24 个 assemble launch 重放一遍作对照。mode 1/2 把垃圾喂给 Mmad/Fixpipe，输出无意义，是测量臂。

| 臂 | 内容 | GM bytes/call | ms（MIN of 5，交错） | GB/s |
|---|---|---:|---:|---:|
| mode 0 | shipped 结构（控制） | 151.0 MB | 0.648 | 233 |
| mode 1 | 只留 GM 流量（无 LoadData/Mmad/Fixpipe） | 100.7 MB | **0.339** | 297 |
| mode 2 | 流量 + Fixpipe store | 151.0 MB | 0.510 | 296 |
| mode 3 | 控制 − P 往还 | 100.7 MB | 0.511 | 197 |
| mode 4 | 只跑 pass 0 | 75.5 MB | 0.327 | 231 |
| 生产 assemble（同进程重放 24 launch） | shipped | 151.0 MB | **0.785** | 192 |

判决算术（本轮脚本自己的输出）：

| 项 | 式子 | 值 |
|---|---|---:|
| (a) GM 发出形态的地板 | mode 1 | **0.339 ms（控制的 52%）** |
| (b) store 侧（两趟各一次 Fixpipe） | mode 2 − mode 1 | +0.171 ms |
| (c) L0 链（LoadData×5 + Mmad + 两次 flag） | mode 0 − mode 2 | +0.138 ms |
| (d) P 往还（含它那 50.3 MB） | mode 0 − mode 3 | +0.137 ms |
| (e) 两趟屏障的每块固定成本 | 2 × mode 4 = 0.655 vs 0.648 | **+0.007 ms（没有）** |
| (f) launch 形态（24 slice） | 生产 − 控制 | +0.137 ms |

判读：

- **瓶颈是小传输本身，不是账面上的字节数**：mode 1 只有 100.7 MB 却要 0.339 ms ⇒ **297 GB/s**，
  是 cube 冷路径边际率（839 GB/s）的 1/3。每 chunk 只有 6 次装载（ave 1.37 KB/次，2 KB + 1 KB + 1 KB
  每 pass），每 chunk 6 次、12288 chunk 共 73728 次调用 ⇒ 4.6 ns/次；若按 839 GB/s 折价，其中
  **0.219 ms 是"传输太小"的价格**（合 3.0 ns/次固定开销）。这正是这次要打的项。
- **两趟结构不花钱**：(e) 说 half-work 臂的两倍与控制臂只差 0.007 ms，也就是两趟之间那道
  `PipeBarrier<PIPE_ALL>` 没有固定成本被摊在块尾。所以"把 assemble 并进 cube"能拿回的只有 (d)
  那 0.137 ms（它顺带省掉 P 的 50.3 MB 往返），**不是** 0.75。
- **可加性自检通过**：(b) + (c) = 0.309 ms = mode 0 − mode 1（0.648 − 0.339），三个候选互不遮蔽，
  控制臂的 0.648 = 0.339（流量）+ 0.171（store）+ 0.138（L0 链）。
- **生产的 0.785 ms 比控制臂贵 0.137**（24 个 launch，≈4.4 µs/launch，与 §11.35 在 cube 上量到的
  slice 价格同值）；§11.34 的 0.928 是 profiler 在整链路竞争下记的，单独重放是 0.785。
- **仍然不在带宽上**：生产绝对率 192 GB/s，把 (a) 的 0.339 打到 cube 价（839 GB/s ⇒ 0.120 ms）
  也就 −0.22 ms；stage 2.535 的 8.7%。而且真到了那一步，AIC（0.55 + cube 1.586）会掉到 AIV 的
  2.144 附近，**stage 变成 AIV 绑**（§11.34 已证明 AIV 那一侧的松弛量没法再用），所以这条线的
  上限就是 ~0.4 ms，不是无穷。
- **合并 DMA 有可测的合法路径**：`Nd2NzParams(ndNum, nValue, dValue, srcNdMatrixStride, srcDValue,
  dstNzC0Stride, dstNzNStride, dstNzMatrixStride)` 允许一次调用搬多张 ND 矩阵，而现在的源布局
  恰好是等跨距的——Lneg 相邻 chunk 相隔 2 KB、Xb 的两个 [M,M] 相隔 4 KB、P 相隔 2 KB，都连续，
  所以"每 pass 每块（NC=4 chunk）的 la 合成一次调用、lb 的 2×NC 个 band 合成一次调用"在布局上
  成立（需要 L1 目标侧的 NZ 跨距按矩阵给）。**当前 kernel 一次都没用 ndNum。**
- 已封死的邻居：把块做肥（`KDA_ASM_NCHUNK = 6/8`）在当时唯一的队列形态下挂核（§11.34，本轮沿用
  NC=4；§11.41 已在显式 buffer 上复测——能跑、无收益）；AIV 侧分支已由 §11.34/§4.7 关闭。

本轮新增：`kernels/v1/k1_solve_assemble_probe.cpp`、`tools/probe_solve_assemble.py`。
生产路径未改。下一步是同一探针上的第二个判决实验——**同样的字节、更少的调用**（先量 1/2/4/8/16/32 KB
的调用尺寸 vs 速率曲线定住固定开销，再把 shipped 的 6 次/chunk 换成按块合并的 2~3 次/chunk），
只有当 mode 1 明显向 839 GB/s 靠拢时才动生产 kernel。

## 11.37. assemble 的小传输被合并掉了：同样 100.7 MB，从 6 次/chunk 压到 2.5 次，生产里位一致、stage −0.178 ms（2026-09-23 深夜）

§11.36 把 assemble 的账算到"小传输自己占 0.339 ms（52%）"上，并指出 `Nd2NzParams` 的 `ndNum` /
`srcNdMatrixStride` / `dstNzMatrixStride` 允许一次调用搬多张 ND 矩阵，而 Lneg / Xb / P 的源布局恰好
都是等跨距的。本轮先做探针，再把赢的那一支落进生产。

**探针**（`kernels/v1/k1_solve_assemble_coalesce_probe.cpp` + `tools/probe_solve_assemble_coalesce.py`，
12288 chunk / grid 3072 / 同进程交错 MIN of 5，每臂字节数固定 100.7 MB，只变调用形态）：

| 臂 | 调用/chunk | 总调用 | ms | GB/s |
|---|---:|---:|---:|---:|
| mode 0 shipped 形态 | 6.0 | 73728 | 0.313 | 321 |
| mode 1 两条 B band 合并（ndNum=KF） | 4.0 | 49152 | 0.252 | 399 |
| mode 2 A 操作数按块合并（ndNum=nch） | 2.5 | 30720 | 0.217 | 465 |
| mode 3 两者都合并（需重排布局的 packed 源） | 1.0 | 12288 | 0.220 | 459 |
| mode 4 整个 block-pass 一次调用（packed 上限） | 0.5 | 6144 | 0.157 | 641 |

判读：**固定开销是真的**（6→2.5 次/chunk 省 0.096 ms，6→4 省 0.061），但 2.5 次之后就没便宜可捡
（mode 2 0.217 ≈ mode 3 0.220），所以 mode 3 需要的重排布局**不值得做**；mode 4 那 0.06 要用 16 KB
大块才拿得到，代价是全部操作数重排，也不做。4.25 ns/次、1.37 KB/次 ⇒ 100.7 MB 里 ~0.31 ms 是
传输形态本身。

**结构比调用数更贵**：把 L0 链和 store 加回来，四臂（同字节、同算术、P/A16 位一致）：

| 结构 | shipped 调用 | 合并后的调用 |
|---|---:|---:|
| 显式 B1 buffer（TBuf，一遍一个 flag） | 0.901 ms | **0.531 ms** |
| shipped 的 per-chunk 队列（EnQue/DeQue） | 0.706 ms | 0.606 ms |

也就是：小调用多的时候队列的 per-chunk 流水值 0.196 ms（0.706 vs 0.901），**但换成大调用之后显式
buffer 反而更快**（0.531 vs 0.606，队列的 per-chunk 同步成了纯开销）。位一致性先于计时：mode 6/8
对 mode 5/7、以及 queue 对 TBuf 的 P 与 A16 全部 IDENTICAL。

**生产**（`kda_solve_assemble` 新增运行期参数 `loadMode`，`api.asm_load_mode()` 读 `KDA_ASM_LOADS`，
每次调用读一次，同一进程可翻臂；0 = shipped，1 = 合并；`tools/probe_solve_assemble_loads.py`）：

| 口径 | mode 0 shipped | mode 1 合并 | Δ |
|---|---:|---:|---:|
| 输出（10.07e7 元素） | — | 0 个不同，max\|d\| 0.000e+00，final state identical | 位一致 |
| e2e（do_bench 中位 of 3） | 10.627 ms（10.620/10.627/10.637） | **10.449**（10.448/10.449/10.450） | **−0.178** |
| ASM（24 launch 重放，MIN of 5） | 0.846 | **0.657** | −0.189（−22%） |
| CUBE（同） | 1.587 | 1.586 | 0 |
| AIC（ASM+CUBE 同流） | 2.619 | **2.310** | −0.309 |
| overlapped（双流无线程事件） | 2.715 | 2.485 | −0.230 |
| **sliced（生产 schedule 重放）** | **2.541** | **2.363** | **−0.178** |
| AIV（wide） | 2.135 | 2.134 | 0（未动） |

- sliced 的 2.541 复现了 §11.34 的 2.535 基线，所以这一列的 −0.178 与 e2e 的 −0.178 是同一笔账：
  **stage 就是 AIC 半边，AIC 少了 0.309，stage 少 0.178**（重叠把那 0.13 的差吃掉）。
- 生产 ASM 0.657 ≈ 转写臂 0.531 + 24 个 launch 的 0.107（§11.35 量的 4.4 µs/launch）＋CrossCoreFlag
  等待，三个数对得上。
- 剩下的账：AIC 2.310 仍高于 AIV 2.134，**stage 还在 AIC 侧**；要翻到 AIV 侧还差 ~0.18 ms，那正好是
  §11.36 的 P 往还（0.137）＋ §11.35 的 cube A16 重读（0.070）同一个量级。
- **队列深度那个理由消失了**：KDA_ASM_NCHUNK 之所以被钉在 4，是 shipped 路径"一个 pass 的装载先发完、
  队列必须 NC 深"（6/8 实测挂核）。显式 buffer 没有这个约束，所以 NC=6/8 现在值得重测——这是后续，
  不是本轮结论。

生产改动：`kernels/v1/k1_solve_assemble.cpp`（两套 L1 形态 + 共用的 per-chunk 链函数，`loadMode` 选路）、
`python/kda_ascendc_v1/api.py`（`asm_load_mode()`，默认 1，随 launch 传参）。公式、dtype、单 head 内
chunk 顺序、同步（CrossCoreFlag / PipeBarrier / 每次 pass 的 MTE2→MTE1）都没动，输出位一致。

**门禁**（默认已切到 `KDA_ASM_LOADS=1`）：`bash tools/run_chunk_matrix.sh` 在 C=16/32/64 三个 leg
全部 PASS（`test_chunk_shape_matrix.py` + `test_stability_gate.py` + `test_c64_gate_overflow.py`）；
新增 `tests/test_solve_assemble_loads.py` 钉住这根旋钮的接线（默认 1、每个 slice 都带、每次调用重读、
两臂输出位一致），C=64 下 3 passed。注意 assemble 只在 C=64 上线（`SOLVE_WIDE_SUBB = 2` 的入场条件），
C=16/32 只会编译它、不会启动它，所以三档矩阵里只有 C=64 那一 leg 真正覆盖新代码路径。

## 11.38. cube 的 A16 常驻落地：转录里值 0.070，生产里 cube −0.066，但 stage 只拿到 −0.019（2026-09-23 深夜）

§11.35 在转录 kernel 上量到"每块把 A16 读一次而不是每 pass 读一次"值 0.070 ms（1.465 → 1.395），
另一半是 L2 命中的那一遍。本轮把它做进生产 kernel：`kda_solve_wu_cube_kernel` 新增运行期参数
`a16Mode`（`api.cube_a16_resident()`，`KDA_CUBE_A16_RESIDENT`，每次调用读一次），0 = shipped
（两遍各读一次，走 `qa` 队列），1 = 每块一次读进 L1 常驻（`TBuf`，`NC*M*K*2` 字节，和它替掉的队列
同字节数），两个 pass 都读它。

**第一次尝试直接挂核**：1 号模式仍然对 `qa` 调了 `AllocTensor` 却从不 `EnQueue`，槽位照样被占满，
第二个 pass 就卡死在队列上（AICore 100% 空转，无输出）——这与 §11.34 记录的 `KDA_ASM_NCHUNK=6/8`
同类：**"AllocTensor 而不 EnQue" 也是一种队列泄漏**。修法是 1 号模式完全不碰 `qa`。

生产 A/B（`tools/probe_solve_cube_a16_resident.py`，三段交错 MIN of 5；本轮运行时 assemble 还是
`KDA_ASM_LOADS=1`）：

| 口径 | mode 0 每 pass 读 | mode 1 常驻 | Δ |
|---|---:|---:|---:|
| 输出 | — | 0/100663296 个不同，max\|d\| 0，final state identical | 位一致 |
| CUBE（24 launch 重放） | 1.577 | **1.511** | −0.066 |
| AIC（ASM+CUBE） | 2.297 | 2.241 | −0.056 |
| sliced（生产 schedule 重放） | 2.346 | 2.327 | **−0.019** |
| e2e | 10.445 | **10.421** | −0.023 |

判读：隔离赢 0.066、stage 只拿 0.019（29%）。这和 §11.34 是同一台机器上的同一件事——sliced 重放把
AIC 的尾巴和 AIV 的重叠吃掉了一部分差值。**默认已切到 1**（严格更好、位一致）。

## 11.39. assemble 的 P 上片（L0C→L1→L0B）落地：拿到 0.107 上限里的 0.096，生产 e2e −0.184，且两半已经配平（2026-09-23 深夜）

§11.36 用"删掉 P 往还"量到它的价格（0.137），§11.37 又把装载改成合并式（stage −0.178）。本轮把
"删掉"变成"搬到片上"——这才是能进生产的形态，因为删掉 P 就没有第二个 Mmad 的操作数。

**探针**（`kernels/v1/k1_solve_assemble_l1p_probe.cpp` + `tools/probe_solve_assemble_l1p.py`，
12288 chunk，装载用 §11.37 的合并形态，三臂只差 P 住哪）：

| 臂 | GM bytes/call | ms | GB/s |
|---|---:|---:|---:|
| mode 0 P 走 GM（控制） | 151.0 MB | 0.490 | 308 |
| mode 1 完全不读不写 P（上限，结果按构造是错的） | 100.7 MB | 0.383 | 263 |
| mode 2 P 走 L0C→L1→L0B | **100.7 MB** | **0.394** | 255 |

**上限 0.107 ms，候选拿到 0.096（90%）**，只比"白送"贵 0.012；且 mode 2 与 mode 0 **位一致**
（A16 0/50331648 个不同，GM 里的 P tile 原封不动——store 真的没了）。

关键在布局：Nd2Nz 的 band-major 源是 `[k-block][n-block]`（这就是 shipped 的
`LoadDataWithTranspose(0, KF*KF, 1, 0, 0)` 能吃四个连续 512 B 分形的理由），而 `CFG_NZ` 的 fixpipe
写出来的是 `[n-block][k-block]`。所以 mode 2 把四个 16×16 分形**按 0、2、1、3 的顺序**塞进 L0B 的
四个槽位，**位一致性是这套映射唯一的验收依据**（错映射不是慢，是错）。

**生产**（`loadMode` 扩成 0/1/2：0 = shipped 队列形态，1 = 合并装载 + P 走 GM，2 = 1 + P 上片；
`kda_solve_assemble` 用同一份 per-chunk 链函数、只在装载与 P 的去处分支）：

| 口径 | mode 0 shipped | mode 1 合并 | mode 2 P 上片 |
|---|---:|---:|---:|
| 输出 | — | 位一致 | **位一致** |
| ASM（24 launch 重放） | 0.866 | 0.665 | **0.513** |
| AIC（ASM+cube） | 2.552 | 2.242 | **2.141** |
| sliced（生产 schedule 重放 = stage） | 2.505 | 2.323 | **2.322** |
| e2e（中位 of 3） | 10.593 | 10.418 | **10.408** |
| AIV（wide） | 2.137 | 2.139 | 2.134 |

- ASM 自己从 0.866 掉到 0.513（−0.353），但 stage 只动 −0.183：**AIC 半边的尾巴被 AIV 的重叠吃掉**，
  这和 §11.34/§11.38 是同一条规律。
- **两半已经配平**：AIC 2.141 对 AIV 2.134，差 0.007 ms。也就是说这条线的账已经结清——此后单独优化
  任何一侧都不会再动 stage，除非**成对**动（要么同时削，要么削完一侧再把另一侧那 0.18 的松弛量拿出来用）。
- mode 1 → mode 2 的 stage 收益只有 0.001 ms（2.323 → 2.322），isolated 是 −0.152；这已经是"用满"的
  信号：AIC 那半边不再是瓶颈。
- 附带事实：mode 2 下 `pmid`（`[c_solve, M, M]`，[1,8192,96,128] 下 25.17 MB/call）既不被写也不被读，
  回收它是后续的清理项（§11.40 已做；这段里说的 50.3 MB 是它的 GM 往还，写 + 读，不是分配量）。

**默认**：`KDA_ASM_LOADS` = 2，`KDA_CUBE_A16_RESIDENT` = 1。这一轮两个改动合起来把 solve stage 从
2.538（本轮起点，§11.37 后的生产值）压到 **2.322 ms（−8.5%）**，e2e 从 10.62 压到 **10.408 ms**，
全部口径位一致。

本轮新增：`kernels/v1/k1_solve_assemble_l1p_probe.cpp`、`tools/probe_solve_assemble_l1p.py`、
`tools/probe_solve_cube_a16_resident.py`、`tests/test_solve_cube_a16_resident.py`，并扩展
`tests/test_solve_assemble_loads.py`（默认 2、三条臂位一致）。生产改动：`k1_solve_assemble.cpp`、
`k1_solve_wu_cube.cpp`、`python/kda_ascendc_v1/api.py`。

**门禁**（默认 `KDA_ASM_LOADS=2` + `KDA_CUBE_A16_RESIDENT=1`）：`bash tools/run_chunk_matrix.sh`
在 C=16/32/64 三个 leg 全部 PASS；`tests/test_solve_assemble_loads.py`（三条臂位一致、默认 2、每 slice
都带、每次调用重读）与新增 `tests/test_solve_cube_a16_resident.py`（两臂位一致、默认 1、每 slice 都带）
在 C=64 下 6 passed。C=16/32 那两 leg 对 cube 的 a16Mode 也生效（单级 solve 同样走这个 kernel），
assemble 仍然只在 C=64 上线。

## 11.40. 清理项落地：mode 2 的 P tile 不再分配（25.17 MB/call 的死内存），位一致、stage/e2e 不动（2026-09-24）

§11.39 记了一笔附带事实：mode 2 下 `pmid` 既不被写也不被读。本轮把它从生产里拿掉。这是**清理**，
不是优化，所以判决口径是"不许有任何变化"，而不是"快了多少"。

**账**（[1,8192,96,128]，C=64：`c_solve` = B * H * nt = 96 * 128 = 12288 个 chunk，`sub` = 32，bf16）：

| 口径 | 字节/call | 说明 |
|---|---:|---|
| `pmid` 的分配 | **25.17 MB** | `[c_solve, sub, sub]` bf16 = 12288 * 1024 * 2 = 24 MiB |
| 它原来的 GM 往还 | 50.3 MB | pass 0 写 + pass 1 读（§11.36/§11.39 里的 50.3 是这个，不是分配量） |

mode 0/1 需要这块 tile（pass 0 的 fixpipe 写 GM、pass 1 的 L0B 从 GM 装），mode 2 两个方向都在片上，
所以**它的存在与否是 mode 的性质，不是调用点的性质**——这正是把它做成"由 `asm_load_mode()` 决定"
而不是"从调用点删掉"的理由。

**改法**（`python/kda_ascendc_v1/api.py`，kernel 一个字节没改）：调用点读 `asm_load_mode()`，>= 2 时
`pmid = None`；`_launch_solve_two_level` 把它按 `None` 传给 `_pack_ptrs`，后者打成空指针。kernel 在
mode 2 下不 deref 它，这不是新承诺——§11.39 的结构就是保证：pass 0 的 fixpipe 目标是 L1 的 `lp`，
pass 1 的 L0B 也从 `lp` 填（`k1_solve_assemble.cpp` 里 `P` 这个 `GlobalTensor` 只出现在 `!onchip`
的两支里）。旋钮每次调用重读，所以探针把 mode 拧回 0/1 时下一次调用会重新分配。

**实测**（`tools/probe_solve_assemble_loads.py`，[1,8192,96,128]，三臂同进程，本轮开头一次跑完）：

| 口径 | mode 0 shipped | mode 1 合并（P 在 GM） | mode 2 P 上片 |
|---|---:|---:|---:|
| 输出 | — | 0/100663296 不同，max\|d\| 0 | **0/100663296 不同，max\|d\| 0** |
| 状态 | — | 相同 | 相同 |
| e2e（中位 of 3） | 10.577 | 10.425 | **10.402** |
| ASM（隔离重放，MIN of 5） | 0.897 | 0.678 | 0.569 |
| CUBE（隔离重放） | 1.506 | 1.511 | 1.508 |
| AIC（ASM + cube） | 2.548 | 2.241 | **2.134** |
| sliced（生产 schedule 重放 = stage） | 2.497 | 2.321 | **2.320** |
| AIV（wide） | 2.125 | 2.125 | 2.132 |

- 对照上一轮（§11.39，同口径）：e2e 10.408 → 10.402，stage 2.322 → 2.320，AIC 2.141 → 2.134——**全在
  噪声里**。清理项该有的样子：省掉的是 25.17 MB/call 的分配和同样多的 HBM 足迹，不是时间。
- **两半仍然配平**：AIC 2.134 对 AIV 2.132，差 0.002 ms。§11.39 的结论不变——这条线再动只能成对动。
- 一个诚实的口径提醒：本轮 ASM 的隔离读数（0.569）比上一轮的 0.513 高 0.056，而 stage/e2e 反向动了
  0.002/0.006。跨进程的 ASM 隔离读数在这个量级上有抖动（这次少分配一块 25 MB，其余 buffer 的地址
  布局就变了，L2 命中随之变），**判决一律用 stage/e2e**，这是 §11.34 立下的规矩。
- 带 sync 的 profile 口径（只做同轮对照）：solve 6.296 / 6.264 / 6.147，total 14.740 / 14.601 / 14.493。

**门禁**：`tests/test_solve_assemble_loads.py` 增一个用例钉住"mode 2 的 P 指针为 0、mode 1/0 非 0"，
与 `tests/test_solve_cube_a16_resident.py` 一起 7 passed（C=64）；`bash tools/run_chunk_matrix.sh` 在
C=16/32/64 三个 leg 全部 PASS。生产改动只有 `python/kda_ascendc_v1/api.py`；
`docs/artifacts/ub_l1_workspace_budget_*.txt` 是旧快照，本轮没有重新生成（它早于工具里 hsnap 那一行，
生成时刻比本轮的这个改动更早）。

## 11.41. KDA_ASM_NCHUNK = 6/8 复测：挂核不回来了，但也没有赢——NC 仍是 4，且隔离读数是地址的、不是内核的（2026-09-24）

§11.34 用"`KDA_ASM_NCHUNK = 6/8` 第一次调用就挂核"把 assemble 的 NC 钉在 4；§11.37 之后生产路径
从 shipped 的 per-chunk L1 队列换成显式 B1 buffer，并在两处文档里把"队列深度这个理由消失了、
6/8 值得重测"记成后续账（`ASCENDC_V1_KERNELS.md`、本文 §11.37）。本轮把账结了。

**方法**（`tools/probe_solve_assemble_nc.py`）：NC 是编译期 define，所以臂 = 同一份 kernel 源码在
NC = 4/6/8/12/16 各编译一份、各自注册成独立 kernel 名（源码改名，defines 前缀走 `api._defines()`
并把该臂的 NC 写进模块属性），这样**一个进程里能交错翻臂**。每个臂先过三关再计时：编译能过、
第一次 launch 能活（挂核就是 §11.34 的失败形态，看门狗是唯一出口）、输出位一致（A16 的 lower-left
块逐元素对齐 NC=4——块变肥只在 block 之间挪活，chunk 内不变）。

**负对照先做**（同一进程、同一 NC=6，只翻 `KDA_ASM_LOADS`）：

| load mode | NC=6 | 说明 |
|---|---|---|
| 0（shipped 队列） | **挂核**（90 s 看门狗超时，AICore 空转） | §11.34 的形态**原样复现**，说明不是驱动/机器变了 |
| 1（显式 buffer，P 在 GM） | 0.478–0.504 ms | 正常 |
| 2（生产，P 上片） | 1.019–1.048 ms | 正常 |

NC=8 同样：mode 2/1 正常、mode 0 挂核。⇒ **挂核是队列形态的性质，不是 NC 的性质**，
§11.37 的推断成立。

**然后量收益**（生产口径，一个进程跑完 NC=4/6/8，每臂换 NC 后重编 `kda_solve_assemble`
并翻转模块属性，`do_bench` 中位 of 3、轮转）：

| 口径 | NC=4 | NC=6 | NC=8 |
|---|---:|---:|---:|
| e2e（3 轮轮转，中位） | **10.389** | **10.390** | **10.392** |
| 三轮原始读数 | 10.388 / 10.389 / 10.400 | 10.385 / 10.390 / 10.391 | 10.389 / 10.392 / 10.393 |
| 输出 | — | 0/100663296 不同 | 0/100663296 不同 |
| （带 sync 的 profile）solve | 5.943 | 5.944 | 5.929 |

**平到噪声底以内**（三轮之间的抖动 0.003–0.012 ms，臂间差 ≤ 0.003）。NC=12/16 同轮也测了：
e2e 11.314 / 11.263（那一轮的机器负载偏高，NC=4 同时读 11.047，比值上同样没有优势），隔离重放
在 NC=4..16 上 0.875–0.888 ms 一路平。**结论：合并传输 + P 上片之后，assemble 在 NC=4 已经不缺
wave，块做肥是免费的，不是赚钱的**——§11.34 想用"降 wave 数"换的那笔账，其前提（wave/依赖受限）
已被 §11.37/§11.39 消掉了。NC 维持 4。

**一个口径教训（本轮最重要的副产物）**：隔离重放读数是**内存地址**的，不是内核的。
同一个 kernel、同一个 NC、同样的字节数，只换操作数张量落在哪块内存：

| 用法 | 隔离重放 |
|---|---:|
| 生产捕获的 24 个 launch（`a16/xb/lneg` 是生产分配） | **0.464 ms** |
| 探针自己 `torch.zeros/randn` 的同样形状张量 | **0.156 ms** |
| 我的 kernel + 生产 args | 0.462 ms |
| 生产 kernel + 我的 args | 0.159 ms |

**3 倍**，且跨 kernel 名可复现——所以 §11.35–§11.40 里 ASM 隔离读数那 0.45–0.9 ms 的散布里有相当
一部分是分配地址/L2 命中，不是内核差异（§11.40 末尾已经记过一次同源抖动，本轮给出机制）。
**判决一律用 e2e/stage**，这条规矩本轮又挣了一次。探针里这一列保留，只用于"活着 + 位一致"。

**门禁**：本轮没动生产 kernel 的任何一个字节（只改了 `k1_solve_assemble.cpp` 的头注释、
`api.py` 里 `ASM_NCHUNK` 的说明）；`bash tools/run_chunk_matrix.sh` 在 C=16/32/64 三个 leg 全部 PASS，
C=64 下 `tests/test_solve_assemble_loads.py` + `tests/test_solve_cube_a16_resident.py` 7 passed。
新增 `tools/probe_solve_assemble_nc.py`。

**另一条环境记录**：第一轮门禁跑在 NPU 0 上时被 `507034 Vector core execution timed out` 打断，
那块卡 `npu-smi` 本来就在 `Alarm` 健康态；换 NPU 3 重跑即过。**本机 NPU 0 不可用于判据**。

## 11.42. 上板 PipeUtilization 的账，以及它判的三条：批量存储赢隔离输 stage、排水链无收益、NZ 写 L1 免费（2026-09-27）

msopprof 的 on-board 采集（`/data/models/Qwen3-4B/kda_msprof_20260924/`，单次 mode-2 launch，
3072 块）第一次把 assemble 的块墙按 pipe 拆开：fixpipe 1.812 us（49.6%）+ mte2 1.201
（32.9%）+ scalar 0.610（16.7%）+ mte1 0.162 + cube 0.098 = **3.884 us** vs 块墙 **3.652 us**，
逐块 98–101% —— **没有两条 pipe 同时忙**，块墙就是这几条链的和。scalar 的 61% 停在
`mte1_stall`（0.249）+ `wait_ib`（0.124）；cube 忙占比 2.7%，与"每块 8 个 32³ Mmad、每个
12.3 ns"的算术量一致（§11.34 的结论不需要修改，只是第一次有了逐 pipe 的实测）。

对着这三条链做了两个探针（`kernels/v1/k1_fixpipe_shape_probe.cpp` +
`tools/probe_fixpipe_shape.py`，与 `k1_solve_assemble_pipe_probe.cpp` +
`tools/probe_solve_assemble_pipe.py`，都在 12288 chunk/3072 块、同进程交错 MIN of 5）：

**一、Fixpipe 形态（判决：隔离赢 0.10，stage 不收，生产仍 2）**

| 臂 | 存储形态 | ms |
|---|---|---:|
| mode 0 生产形态 | 4x NZ→L1 + 4x 跨行距（stride=PC）行主序→GM = 8 次 | 0.373 |
| mode 1 批量（`ndNum=4`） | 4x NZ + 1 次跨行距批量 = 5 次 | **0.278** |
| mode 2 8x NZ→L1 | 8 次 | 0.184 |
| mode 3 8x 连续 2 KB 行主序（stride=M） | 8 次 | 0.358 |
| mode 4 8x 跨行距行主序（stride=PC） | 8 次 | 0.556 |
| mode 5 无存储 | — | 0.174 |

- **批量合法**：`FixpipeParamsV220` 的 `ndNum`/`srcNdStride`/`dstNdStride` 走 NZ2ND；扫描
  srcNd = 2/4/8/16/32，**只有 4（= MM/256，1 KB 分形单位）位一致**（0/50331648），其余
  都在 75% 写域外或错块——写域校验要把"kernel 不写的 75%"切掉再比。部署为生产 kernel 的
  **mode 3**（`KDA_ASM_LOADS=3`，默认仍 2；`CAN_BATCH_STORE = (MM % 256) == 0` 守卫 M=8 的
  compile-only 档）。
- **跨行距是真实的 2 倍代价**（mode 4 0.556 vs mode 3 0.358），而 NZ→L1 几乎免费（+0.010）。
  所以 Fixpipe 那 1.812 us 里主要不是"每次调用的固定价"，是**跨行距写 GM 的那条路径本身**。
- 生产判决（`tools/probe_solve_assemble_loads.py`，四臂轮转，正序+反序两轮）：
  ASM 隔离 0.572→**0.471**、AIC 2.146→**2.008**，但 **sliced 2.320→2.335、e2e +0.020
  （反序 +0.031）**——隔离赢的那 0.10 全被 stage 吃掉还倒亏。机制：批量存储把 4 次 store
  从"与 Mmad 交错"变成 pass 1 末尾的一条串行尾，而 sliced 的切片内 assemble→cube 是串行的
  （cube 要等 A16），尾巴直接进关键路径。**判：不启用**（旋钮保留，测试钉住 mode 3 可达、
  位一致、不分配 P tile）。
- 顺带一条负结果：`k1_solve_assemble_pipe_probe.cpp` 把每 unit 的排水链拆了三层
  （hoist pass-1 la / phases 全填充-全 Mmad-全 store / ping-pong M_FIX 交错 / depth-1 装载
  流水，共 5 个变体），**全部落在 ±0.015 ms**（hoist −0.015 最大，在噪声底内）——那 3.884 us
  的串行和**不是 flag 编排造成的**，改 flag 结构没有出路；cube 的"等待"就是这么分工的
  算术量（每块 98 ns）。

## 11.43. K2 的 v_new 重排拆开量：gather 与转置各占一半，TransDataTo5HD 替换更慢（2026-09-27）

§11.25 把 K2 的 v_new 重排（16 次跨步 gather + 16 次 16x16 `Transpose` + packed `Vt`
store）的总价记成 V4b 的 −0.214（含当时未守卫的 `V` store）。`V` 的守卫已经落地（−0.095），
本轮把剩下的结构项拆开（`tools/probe_k2_vt_split.py`，[1,8192,96,128]/C=64、24 块、
同进程 MIN of 11 交错，全部在 `V` store 已守卫的基线上）：

| 变体 | min_ms | Δ | 含义 |
|---|---:|---:|---|
| V0 stock（重编译） | 4.002 | +0.006 | 噪声底 |
| V6 无 gather、无转置（§11.25 地板） | 3.805 | **−0.191** | 整段的上限 |
| V5a 保 gather、去转置 | 3.915 | −0.082 | 转置的边际 ≈ 0.11 |
| V5b 一次整块拷贝替 gather、保转置 | 3.864 | −0.132 | gather 的边际 ≈ 0.06–0.14 |
| **V7 `TransDataTo5HD` 替 `Transpose`** | 4.150 | **+0.154** | **判死** |

判读：

- 这 0.19 ms 由两半组成，且**两半都不小**：gather 的 16 次 32 B/行小拷贝约 0.11–0.14，
  16 次 `vtranspose` 约 0.06–0.09（配对不同、有重叠，精确分解受限于消融的非可加性）。
- **V7 是唯一"合法替换"候选，实测比被替换的 16 次原语慢 0.154 ms**——`TransDataTo5HD`
  在这个尺寸（16x16 bf16、`NCHW_CONV_ADDR_LIST_SIZE` 的地址表构建在标量上）不划算。
  它的数值未再校验（速度先判死）。
- 剩下的合法形态只有"把转置搬到 AIC 的 `LoadDataWithTranspose`、省掉 packed `Vt` 与这
  16 次转置"（§11.7 记过的方向）——那要用 AIC 侧描述符换 AIV 侧指令，且 K2 现代码里 `Vt`
  的 packed 序正是 §11.11 用 0.82 ms 的 store 换来的，重开需要整段重排 + 两个消费点
  （stage 1/3 的 A/B 操作数）同步改。本轮不落地，记入待办。

探针入库：`tools/probe_k2_vt_split.py`（原一次性脚本 `tools/probe_k2_vt.py` 也一并入库）。

## 11.44. pre_gram 的 MTE3_V marker 在今天的代码上重验：仍然负载承载，UB 依然放不下（2026-09-27）

§11.25 的 D2 在**当时**的 kernel 上删掉 pass 边界的 `MTE3_V` 自配对 marker 后量到 −0.230 ms，
但逐位 diff 显示 Rv/U/Vnew/VnewT 动 0.32、L/Aqk/A16/A32 动 ~0.05，所以判"负载承载"并冻结
（恢复收益需要把跨 pass 复用的 4 个 tile 双缓冲 = 16 KB，而 UB 余量只有 4 KB）。
此后 kernel 改过多处（mask 提到 per-block、bT0 结构、qgin/qgmk/qgout 等），**这条冻结的理由
需要复验**。

本轮在同一探针上重跑（`tools/probe_pg_marker.py`，[1,1024,16]/C=64、14 个中间量逐位 diff）：

- 删 marker 后 **Rv/Vnew/VnewT/U 依旧动 3.203e-01、L 4.949e-02、Aqk 1.364e-02、A16/A32
  4.932e-02**——与 §11.25 表里的数字**逐位相同**，两端的失配没有变；
- UB 账本（`tools/gen_ub_l1_budget.py` 的 `pre_gram_ub(64)`）：**189792 / 196608 B**，即
  余 6.8 KB，而双缓冲那 4 个 tile 要 16 KB——**赤字 ~9.5 KB，结论不变**。

**判决：维持冻结。** 恢复这 0.230 需要先腾 ~9.5 KB UB（唯一的大块 bT0 已被 §11.25 证明切不了：
cumsum 需要整 chunk 顺序链、kg 需要第 M-1 行、XBAND 需要第 MID0+BS 行），或者把 marker 换成
别的同步形态——两条路都要重跑数值 gate，收益 1% 量级。

**本轮"把方案都试一遍"的收束表**（全部实测量，探针已入库）：

| 项 | 隔离读数 | stage/e2e | 判决 | 阻塞 |
|---|---:|---:|---|---|
| solve A16 批量存储（`ndNum`） | −0.095（ASM 0.572→0.471） | **+0.020 / +0.031** | **不启用**（mode 3 保留） | 批量 store 成为 pass 1 串行尾 |
| solve 排水链 6 变体（hoist/phases/interleave/深度1/**全阶段滚动**） | ±0.015 | — | **不改** | flag 编排不是成本（§11.45） |
| K2 gather + 16 转置 | −0.191（V6 地板）/ −0.152（本轮复测） | — | 拆解完成 | 合法形态需要 AIC 侧重排（记录待办） |
| K2 `TransDataTo5HD` 替换转置 | **+0.154** | — | **判死** | 更慢 |
| pre_gram `MTE3_V` marker | −0.230（§11.25） | — | **维持冻结** | UB 缺 9.5 KB + 负载承载复验 |
| pre_gram `FL_DONE` 协议加深 | 0.17（§11.20） | — | **不动** | 两次挂死史，风险 >> 2% |
| solve 切片深度（48/96） | — | 更差（§11.9 已扫） | **已冻结** | — |
| **stage 配平的墙** | AIC 2.134 vs AIV 2.132 | — | **单侧改动不再动 stage** | 除非成对削（§11.39/§4.10） |

## 11.45. 单个计算内部的"边算边传"（双缓冲）也判死：第 6 个排水变体全阶段滚动落在噪声里，cube 的 2.7% 是工作量不是调度（2026-09-28）

问题（用户侧）：能不能在**一个 pass 内部**做通算掩盖（参考 Ascend C 双缓冲实践：copy-in(k+1)
压在 compute(k) 下），让 cube 的等待被别条 pipe 的搬运盖住。§11.42 的 5 个变体都不是这件事：
phases 等的是整个 pass 的 lift，interleave 只让 FIX 压 M（上限 98 ns），depth-1 只滚 MTE2。

`k1_solve_assemble_pipe_probe.cpp` 增加 **mode 6 rolling**：链上四个阶段各提前一个 chunk 发——
`fill(ch+2)`（MTE2→L1）/`lift(ch+1)`（L1→L0A/L0B）/`mad(ch)`（L0C）/`fix(ch-1)`（L0C→L1/GM）
在同一轮循环里按序发，稳态下四条 pipe 各持一个不同 chunk。MTE1_M 与 M_FIX 各用 ping-pong id
（每个 id ≤1 outstanding set），MTE2_MTE1 仍是 mode 5 的 per-chunk 对；prologue/稳态/收尾由同一
body 的边界判断覆盖；pass 边界保留原 barrier（P 的 FIX→MTE1 依赖不变）。

（12288 chunk / grid 3072 / CHUNK=64 / ASM_NCHUNK=4 / 同进程交错 MIN of 5；每臂先零填 A16 再逐位比对）

| arm | ms | vs control |
|---|---:|---:|
| mode 0 control（生产链） | 0.398 | +0.000 |
| mode 1 hoist | 0.384 | −0.014 |
| mode 2 phases + hoist | 0.407 | +0.010 |
| mode 3 interleave | 0.404 | +0.006 |
| mode 4 phases, no hoist | 0.417 | +0.020 |
| mode 5 depth-1 | 0.409 | +0.012 |
| **mode 6 rolling** | 0.411 | **+0.013** |

A16 全臂 IDENTICAL（0/50331648）。**7 个臂全在 ±0.015 ms（≤3.8%）**，mode 6 一截都没回收。

判读（回答"cube 能不能靠边算边传提高利用率"）：

- **算术上限先摆在这**：cube 每块 98 ns（8 个 32³ Mmad × 12.25 ns = 2.7%），这是"能被掩盖的
  compute"的全部；fix 1.812 是 store 路径、mte2 1.201 是 load 路径、scalar 0.610 是搬运/等待指令
  的发射——它们本身是数据通路在忙，不是"等算术"。参考页成立的前提（copy 与 compute 同量级、
  交替遮蔽）在这里是 **compute : copy ≈ 1 : 37**。
- 若"3.884 = 3.652、没有两条 pipe 同时忙"是排水结构造成的调度损失，mode 6 这种最大重叠形态
  至少该回收 mte2 或 cube 的一截，实测 **0**。所以块墙是**数据通路自己的账**，不是 flag 编排的账。
- 生产路径**曾经就是参考页那个形态**：per-chunk 队列双缓冲（EnQue/DeQue）。§11.37 已测：小调用
  时它赢 0.196（0.706 vs 0.901），换成大调用后反而输 0.075（0.606 vs 0.531）——现在的显式 B1
  buffer + 装载合并就是它的替代品，且位一致。
- 提升 cube 利用率只剩两条真路：**(a) 给 cube 更多有用的活**（每字节搬运配更多 Mmad）；
  **(b) 减少搬运**——后者正是已落地收益的来源（装载合并 stage −0.178、P 上片 e2e −0.184），
  且两半已配平（AIC 2.134 / AIV 2.132），单边再动不能动 stage。

**上板核对**（同日 `msprof op --aic-metrics=PipeUtilization`，mode 0/5/6 各一次单独采集，
归档 `/data/models/Qwen3-4B/kda_msprof_20260928_rolling/`，INDEX.md 与 SUMMARY.txt 同目录；
`tools/parse_msop_pipeutilization.py` 本轮入库，SUMMARY 就是它的输出）：

| 采集 | 块墙 | fixpipe | mte2 | mte1 | cube | scalar | 五 pipe 和 / 块墙 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 生产 2026-09-24（合并装载，排水链） | 3.652 | 1.812 | 1.201 | 0.162 | 0.098 | 0.610 | 1.063 |
| mode 0 control | 3.582 | 1.816 | 1.205 | 0.162 | 0.098 | 0.550 | 1.069 |
| mode 5 depth-1 | 3.837 | 1.848 | **1.769** | 0.217 | **0.181** | 0.583 | 1.198 |
| mode 6 rolling | 3.829 | 1.845 | **1.773** | 0.218 | **0.182** | 0.702 | **1.233** |

`五 pipe 和 / 块墙` 就是"平均同时在忙的 pipe 数"：1.06–1.07 说排水链确实把 pipe 压成近似串行，
mode 6 把它抬到 **1.23**——**滚动调度在硅上真的换来了并行**（这是本页参考材料"不同指令队列可并行"
的正面证据），但它仍然没有缩短块墙，原因在这一列的代价里：

- mte2 的 +0.57 us 与 cube 的 +0.08 **不是滚动的账**：mode 5 一样有（它只滚 MTE2），
  是"滚动所需的 per-chunk 装载"的账——每块 12 次 `DataCopy` 而不是 6 次，而 §11.37 已测同一批字节
  小调用 321 GB/s、合并 465 GB/s。
- mode 6 相对 mode 5 多出来的才是滚动本身：scalar +0.12（每个 chunk 多了 ping-pong 的等待/发射）、
  并行度 +3.5 个点、块墙 3.837 → 3.829（没有变化）。launch 级（交错 MIN of 5）mode 5 +0.012、
  mode 6 +0.013，7 个臂全在 ±0.015 ms。
- 上限的算术没变：能被藏的 compute 只有 cube 的 98 ns/块（2.7%），而 fix 1.812 + mte2 1.201 +
  scalar 0.610 本身就是墙；**用"把数据通路的活切碎"换来的并行，付的价比重叠收回的多**。

复现：`KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_solve_assemble_pipe.py`
（探针与脚本已入库；设备为 Ascend910B3，20 个 cube 核，0.398 ms ↔ 2.59 µs/块）；
上板：`KDA_MSOPP_KERNEL=kda_solve_assemble_pipe_probe KDA_MSOPP_MODE=6 msprof op ...`
（`tools/msop/run_msop.py` 本轮扩了 `GEOMETRY` 与 `KDA_MSOPP_MODE`，见归档 INDEX.md）。

## 11.46. 探针：跨 launch 字节的边际价格是 1.43 ms/GB，账本用的 0.2 错了 7 倍，Level 2/3 的排序要重排（2026-09-28）

问题（承接"还有什么办法缩端到端时间"）：账本里同一件事挂着两个价，从来没有对过账。
§11.12 量到 **0.2 ms/GB**，但那量的是"同一次 launch 里后面的 block 复读"（L2 命中）；
§11.25 路线 C 量到跨 launch 交接 **2.01 GB → 2.104 ms = 1.047 ms/GB**，已经贴近这台机器
1165 GB/s 的拷贝屋顶。§11.28/§11.29 那两张候选表用的是便宜的那个：Level 2 省 0.6 GB 记
0.12 ms、Level 3 省 0.4 GB 记 0.08 ms，两级合计 0.2 ms，于是结论是"不值得为 0.2 ms 重排跨
stage 交接"。**而 Level 2/3 省的恰恰是跨 launch 的一趟 GM——用错了价。** 这一节把价直接量出来。

### 1. 方法：探针就是那个 kernel，唯一变量是 RHS 的地址

选全流水里最大的一块跨 launch 读：`kda_solve_wu_cube_kernel` 读 rk/rv，两个张量都是更早的
launch（pre_gram）写的，合计 402.65 MB/调用，是 L2 的 25 倍。`kernels/v1/k1_solve_cube_rhs_probe.cpp`
是 `k1_solve_wu_cube.cpp` 的转录（A16 常驻，即生产默认 `KDA_CUBE_A16_RESIDENT=1`），
**除了 RHS 的地址以外每条指令都一样**（qa/qb 队列协议、两次 LoadData 过片、Mmad、Fixpipe、
L0A/L0B/L0C 槽位算术、InitBuffer 账全同）。四臂：

| mode | 含义 | RHS 地址 |
|---|---|---|
| 0 | 控制（= 生产） | `rhs[c0 + ch]`，冷 |
| 1 | L2 热 | `rhs[(c0 % hot) + ch]`，调用数/形状/指令流完全相同，全网格只读一个 `hot` 窗口 |
| 2 | 地板 | RHS 的 `DataCopy` 全删（L1 槽仍 Alloc/EnQue/DeQue 并跨到 L0B，队列协议与转置不变） |
| 3 | 半冷 | pass 0 冷、pass 1 热（线性点） |

`hot` 是**实参不是常量**，所以窗口大小可以在同一个二进制上扫，用来证明量到的是字节而不是窗口。
1/2/3 臂算的是垃圾（故意的），所以这一轮**没有位一致 gate**——口径同
`k1_solve_wu_cube_a16_probe.cpp` 的 mode 2；控制臂靠在**同进程回放生产 kernel 的 captured
launch**（24 个）挂钩现实。

### 2. 数据（12288 chunk / grid 6144 / CHUNK=64 / NC=2 / 每块 16 次 Nd2Nz / 同进程交错 MIN of 5）

device 3（device 0 本轮 `npu-smi` Health = Alarm，按 §11.41 的规则不进结论）：

| 臂 | 冷读 GB/调用 | ms | ms/GB(冷) | vs 控制 |
|---|---:|---:|---:|---:|
| mode 0 控制（生产地址） | 0.403 | **1.422** | 1.427 | +0.000 |
| mode 1 热，窗口 32 chunk（1.0 MB） | 0 | **0.847** | — | −0.575 |
| mode 1 热，窗口 128 chunk（4.2 MB） | 0 | 0.862 | — | −0.560 |
| mode 1 热，窗口 512 chunk（16.8 MB） | 0 | 0.913 | — | −0.510 |
| mode 3 半冷（只 pass 0 冷） | 0.201 | 1.127 | 1.386 | −0.296 |
| mode 2 地板（无 RHS 调用） | 0 | **0.615** | — | −0.807 |
| 生产 cube（回放，24 launch） | 0.403 | 1.515 | 1.657 | +0.092 |

三条自证（这一轮的读数之所以能用，全靠这三条）：

1. **窗口不敏感**：窗口从 32 扩到 512（16 倍，1.0 → 16.8 MB，已经大于任何 L2 分片），热臂只慢
   0.066 ms，价格仍是 1.27 ms/GB。所以 (0−1) 量的是字节，不是"恰好塞进某个窗口"。窗口也不是
   §10 那种单地址病态：16 个 chunk 基址 × KF 个 4 KB 颗粒。
2. **线性**：冷的一半 +0.296、热的一半 +0.279，差 0.017 ms（噪声底 ±0.014）。价格是**按字节**
   计的，不是按调用计的——所以它可以拿去乘别的字节数。
3. **转录可信**：控制臂 1.422 对生产回放 1.515，差 +0.092。生产是 24 个 launch（overlap 切片），
   控制臂是 1 个；按 §11.35 量到的 4.4 µs/launch，23 × 4.4 µs = 0.10 ms。**差值就是 launch 数，
   不是内核差异。**

### 3. 判决：1.427 ms/GB（701 GB/s），并且它比拷贝屋顶还贵是有原因的

- 是账本 Level 2/3 用的 L2 价（0.2 ms/GB）的 **7.1 倍**；
- 比 §11.25 路线 C 的 1.047 ms/GB 还贵 **36%**；
- 比纯拷贝屋顶（1165 GB/s ⇒ 0.858 ms/GB）还贵 **66%**。

这 66% 不是矛盾，是**边际价的定义**：它是在该 kernel 自己的 0.403 GB Fixpipe 写同时在跑的条件下
量出来的，付的是读写混合 + 4 KB 颗粒突发长度的价。旁证是 §11.29 独立量到的**写**价
（405.80 MB ⇒ −0.164 ms = 0.404 ms/GB）比这里的读价便宜 3.5 倍——同一个方向：这台机器上
"读"比"写"贵，而账本的 0.2 ms/GB 两个都不是。

**拆解**：RHS 在控制臂上一共花 1.422 − 0.615 = **0.807 ms**，其中 **0.575 ms 是字节（71%）、
0.232 ms 是 16 次调用本身**（≈47 ns/调用/核）。字节是大头，所以"减少调用数"（§11.37 那一类
合并）在这个 kernel 上最多拿 0.232 的一部分，而"删字节"能拿 0.575。

### 4. 复价：那两张候选表按 1.43 ms/GB 重算

| 方向 | 账本价（L2 0.2 ms/GB） | 按本轮实测重算 | 倍数 |
|---|---:|---:|---:|
| Level 2 `pre_gram->solve` window 流式（0.6 GB） | 0.12 ms | 0.6 × (1.43 − 0.2) ≈ **0.74 ms**；若连写回一起省，+0.6 × 0.40 ≈ **0.98 ms** | 6–8× |
| Level 3 `solve->K2` window 流式（0.4 GB） | 0.08 ms | 0.4 × 1.23 ≈ **0.50 ms**（读腿） | ≤6× |
| cube RHS 装载合并（16 次 → 4 次整 tile Nd2Nz，A16 路径已经是这个形态） | 未标价 | 调用腿上限 0.232 × 3/4 ≈ **0.17 ms**（隔离），突发从 4 KB 变 16 KB 还可能再动字节腿 | 新候选 |

### 5. 但两道墙没变，必须一起说

1. **stage 配平墙**（§11.39/§11.40）：AIC 2.134 对 AIV 2.132，差 0.002。cube 属 AIC 半边，
   **单边削它不动 stage**。§11.38 给过折扣率：隔离 0.070 → 生产 cube −0.066 → **stage 只 −0.019**
   （缩水 3.7 倍）。所以上面那个 0.17 ms 的 RHS 合并，落到 stage 上按同一折扣只剩 ~0.05 ms，
   除非与 AIV 半边成对削。
2. **Level 2 不吃这道墙**，这是它比 cube 侧候选值钱的结构性原因：它削的是 pre_gram 的**写**
   加 solve **两半**的读（`L masked` 由 wide/AIV 与 assemble/AIC 读，`Rk/Rv` 由 cube/AIC 读），
   是流水线级的字节删除，不落在任何一侧的配平里。而全流水 9.27 GB ÷ 10.40 ms = **0.89 GB/ms
   = 拷贝屋顶的 77%**（pre_gram 一家 86%）：**在已经贴近屋顶的流水里，唯一还能拿的量级就是删
   字节；而删字节刚刚被证明比账本以为的贵 7 倍——也就是值钱 7 倍。**

### 6. 本轮没有回答的那一条（下一个便宜探针）

热臂证明的是"**同一次 launch 内**被 6144 个 block 复读的小窗口确实常驻 L2"（价格 → ~0）。
Level 2/3 需要的是更强的那一条：**一个 2 槽 window ring 能不能活过一次 launch 边界**——生产者在
launch N 写、消费者在 launch N+1 只读一次，没有 launch 内复用来兜底。§11.25 路线 C 在 2.01 GB
上量到 1.047 ms/GB（即 L2 没帮上忙），但那个尺寸本来就装不下，**对 ring 尺寸什么也没说**。
所以复价后的 0.74 ms 是**上界**，准入前要先用同一个转录套路量这一条：两个 launch、一个小 ring、
对照一个等字节的大冷缓冲。这条探针的成本与本轮同量级（一个 .cpp + 一个 tools 脚本）。

**这条探针本轮做了三版，仪器始终不合格，问题仍然开放——但排除法把范围收得很窄了。**
`kernels/v1/k1_l2_survival_probe.cpp` + `tools/probe_l2_survival.py`：固定 grid 6144、固定
4 KB 颗粒（就是 cube 的 Nd2Nz 突发长度），每块拥有同样 8 个 tile 的跨度，`ntile` 决定碰几个，
于是 footprint 从 25.2 MB 扫到 201.3 MB 而每块指令形状与地址布局不变；三臂是消费者单独 /
生产者单独 / 两者背靠背（后者就是 Level 2/3 的"写在 launch N、读在 launch N+1"），并减掉
`ntile=0` 的 launch 地板（0.219 ms）。

| 版本 | 收尾存储 | +201 MB footprint 换来的读腿 |
|---|---|---:|
| 1 | 每槽 32 B（合计 1.57 MB） | +0.003 ms |
| 2 | 整块 32 KB tile 进 4 MB ring（合计 201 MB） | +0.002 ms |
| 3 | 同 2，外加实参回显 | +0.002 ms |

201 MB 的冷读按本轮的 1.427 ms/GB 至少要 0.29 ms，实测差了 **140 倍**——所以这不是"便宜"，
是仪器没在量。第 3 版把两个最可能的解释都排掉了：

- **不是实参没送到。** block 0 把自己收到的 `ntile`/`mode` 按位置编码盖进 `out` 的尾部
  （bf16 没有 float 标量的 `Duplicate`，所以用"往 `echo + v*16` 拷 16 个元素"来编码 v），
  host 读回来是精确的 `(8,1) (3,1) (8,0) (1,3)`，**四个组合全对**。
- **也不只是加载被优化掉。** 收尾存储自己那 201 MB 同样只值 **0.000 ms**：地板在 1.57 MB
  收尾时是 0.220 ms，换成 201 MB 收尾是 0.219 ms。**这个 kernel 里的 GM 流量两个方向都不计价。**

顺带留档两个负结果，都比结论本身更省后来人的时间：

- 把"生产者有没有把 span 写回去"当回读通道是**不可靠**的。它给出的图案在每个 `ntile` 上都乱
  （ntile=1..4 与 7 全 0、5 命中 tile 4、6 命中 tile 5、8 命中 tile 6/7），既不匹配前缀图案，
  也不匹配任何位移/交换假设，所以它连"两个 int32 被交换了"都证伪不了。**回读要读实参本身，
  不要读实参的后果。**
- 第一版还漏了 `torch.npu.synchronize()`：torch_npu 不保证自己的 elementwise op 与裸
  `aclrtLaunchKernel` 定序（`api.py` 在 eye-tile 缓存那里专门写过这条），于是回读到假阴性。

脚本现在**自带守卫**：最大 footprint 的读腿不到冷价预测的 25% 就打印 `INSTRUMENT INVALID`
并拒绝给判决——第一版正是把这条直线读成了"L2 不存活"，那会是一个被写进账本的错误结论。

**下一个仪器不要再写独立微基准。** §11.46 的 cube 转录是**已知能搬真字节**的（三臂
0.615 / 0.847 / 1.422 ms，且随字节单调），存活臂应该长在那里：先用一个 launch 写出一个 ring
大小的 RHS 窗口，再让 cube 转录把它当 RHS 读，对照同一个窗口冷读。

**所以 §11.46 第 4 节那个 0.74 ms 仍然是上界，不是可兑现值**；兑现前必须先量这一条。

净结论：§11.28/§11.29 把 Level 2/3 排到最后，理由是"两级合计 0.2 ms，不值得动跨 launch 交接
协议与同步审计"。**这个理由的价错了 7 倍。** 准入成本没有变便宜，但收益侧从 0.2 变成
0.74–1.0 ms（e2e 的 7–10%），比 §11.37 + §11.39 两轮加起来落地的 0.36 ms 还大，
**顺序应当重排**。

复现：`KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_solve_cube_rhs.py`
（探针 `kernels/v1/k1_solve_cube_rhs_probe.cpp` 与脚本本轮入库；跑之前先看 `npu-smi` 的 Health 列）。
第 6 节那个不合格的仪器同样入库（`kernels/v1/k1_l2_survival_probe.cpp` +
`tools/probe_l2_survival.py`，跑法同上）：**留着是为了让守卫和三版的读数可复现**，它现在只会打印
`INSTRUMENT INVALID`，不会给判决。

## 11.47. 探针：L2 能带着交接活过 launch 边界——17 MB 的 ring 回收了 89–100% 的跨 launch 读价，§11.46 的 0.74 ms 从上界变成可兑现（2026-09-28）

§11.46 把跨 launch 读复价到 **1.427 ms/GB**，Level 2/3 因此从账本的 0.20 ms 变成 0.74 + 0.50 ms。
但那是**上界**：只有"2 槽 window ring 写在 launch N、读在 launch N+1 时被 L2 接住"才兑现得了，
而 §11.46 的热臂证不了这条——它是 launch **内** 6144 个 block 复读，一次 miss 摊到 ~384 次命中，
就算每个 launch 边界都刷 L2 它也照样看起来免费。本轮把这条量了。

### 1. 第一版仪器（独立微基准）三版全废，废因值得留档

`kernels/v1/k1_l2_survival_probe.cpp` + `tools/probe_l2_survival.py`：固定 grid 6144、固定 4 KB
颗粒，每块拥有同样 8 个 tile 的跨度，`ntile` 决定碰几个，footprint 扫 25.2 → 201.3 MB。

| 版本 | 收尾存储 | +201 MB footprint 换来的读腿 |
|---|---|---:|
| 1 | 每槽 32 B（合计 1.57 MB） | +0.003 ms |
| 2 | 整块 32 KB tile 进 4 MB ring（合计 201 MB） | +0.002 ms |
| 3 | 同 2，外加实参回显 | +0.002 ms |

201 MB 冷读按 1.427 ms/GB 至少要 0.29 ms，实测差 **140 倍**。第 3 版把两个最可能的解释都排掉：

- **不是实参没送到**：block 0 把收到的 `ntile`/`mode` 按位置编码盖进 `out` 尾部（bf16 没有
  float 标量的 `Duplicate`，所以用"往 `echo + v*16` 拷 16 个元素"编码 v），host 读回精确是
  `(8,1) (3,1) (8,0) (1,3)`，四组全对。
- **也不只是加载被优化掉**：收尾存储自己那 201 MB 同样只值 **0.000 ms**（地板在 1.57 MB 收尾时
  0.220、换成 201 MB 收尾 0.219）。**那个 kernel 里 GM 流量两个方向都不计价**，原因未定位。

两条方法学教训比结论更省后来人的时间：

1. **回读要读实参本身，不要读实参的后果。** 早一版用"生产者有没有把 span 写回去"当通道，图案在
   每个 `ntile` 上都乱（1..4 与 7 全 0、5 命中 tile 4、6 命中 tile 5、8 命中 tile 6/7），既不匹配
   前缀图案也不匹配任何位移/交换假设，**连"两个 int32 被交换了"都证伪不了**。
2. 裸 `aclrtLaunchKernel` 之前必须 `torch.npu.synchronize()` 发布 torch 自己的写（`api.py` 在
   eye-tile 缓存那里专门写过这条），第一版漏了，读到假阴性。

脚本现在自带守卫（最大 footprint 的读腿不到冷价预测的 25% 就打印 `INSTRUMENT INVALID` 并拒绝
判决）。第一版正是把这条直线读成了"L2 不存活"——那会是一个被写进账本的错误结论。

### 2. 换仪器：消费者用 §11.46 那个**已知能搬真字节**的 cube 转录

`tools/probe_ring_survival.py`，**没有新 kernel**：

- 消费者 = `kda_solve_cube_rhs_probe` mode 0（三臂 0.615/0.847/1.422 ms、随字节单调，仪器已被证明）；
- 生产者 / 驱逐器 = torch 的 elementwise kernel `torch.mul(..., out=)`。**不用 `fill_`/`copy_`**，
  那两个可能降级成 memset/memcpy，走的是本流水里任何生产者都不走的路径；
- 臂：`hot = t[P;C] − t[P]`，`cold = t[P;E;C] − t[P;E]`，**两臂带同样的生产者前缀**，唯一差别是
  消费者跑的时候 ring 还在不在 L2；
- ring 尺寸 R ∈ {512,1024,2048,4096,8192} chunk ⇒ 16.8 / 33.6 / 67.1 / 134.2 / 268.4 MB，
  **knee 由测量给出，不假设 L2 容量**；
- 每次计时里消费者跑 K=4 次，所以热臂比 Level 2/3 的真实形态**更苛刻**（ring 要活过 4 次消费 +
  4×ring 的 W/U 写）。下面的回收率对 Level 2/3 是**下界**。

（device 3，交错 MIN of 3 × 4 轮；两次独立运行并列）

| ring | 每次计时读量 | hot leg | cold leg | 回收 ms/GB（run1 / run2） | 占冷读价 1.427 |
|---|---:|---:|---:|---:|---:|
| 16.8 MB | 0.067 GB | 0.146 / 0.154 | 0.242 / 0.239 | **1.427 / 1.272** | **100% / 89%** |
| 33.6 MB | 0.134 GB | 0.276 / 0.280 | 0.338 / 0.333 | 0.392 / 0.414 | 27% / 29% |
| 67.1 MB | 0.268 GB | 0.568 / 0.562 | 0.666 / 0.677 | 0.428 / 0.411 | 30% / 29% |
| 134.2 MB | 0.537 GB | 1.798 / 1.784 | 1.900 / 1.869 | 0.159 / 0.176 | 11% / 12% |
| 268.4 MB | 1.074 GB | 3.634 / 3.600 | 3.691 / 3.676 | 0.071 / 0.006 | 5% / 0% |

### 3. 仪器自证：两个方向量到同一个价

**最小 ring 回收的 1.272–1.427 ms/GB，与 §11.46 在另一个 kernel、另一套臂上独立量到的冷读边际价
1.427 ms/GB 对上（89–100%）。** "把读变冷要花多少"和"把读变热能省多少"给出同一个数，所以这条价
不是某个探针的假象。另有两条守卫，脚本两条都查、任一不过就拒绝判决：回收量不能超过冷价
（1.272 ≤ 1.427 ✓），且必须随 ring 增大衰减（1.272 → 0.006，208 倍 ✓）。

**绝对 ms/GB 这里是 2.1–3.6 而不是 1.43，这是对的**：这两条腿是**整个消费者**的账（含它自己的
W/U 写 4×ring、A16 读、4 次 launch 开销），不是 RHS 的边际账。判决只用**两臂之差**——那是唯一随
L2 状态变化的量。第一版脚本拿整腿的绝对价去比边际价，误报了 `INSTRUMENT INVALID`，已改成用差值比。

### 4. 判决，以及它给 Level 2 加的硬约束

- **L2 能带着交接活过 launch 边界，但有尺寸上限**：live ring ≲ 17 MB 时回收 89–100%，34–67 MB 掉到
  ~29%，134 MB 只剩 ~12%，268 MB 归零。所以这台机器上"能被交接复用的 L2"实测是 **16–32 MB 量级**
  ——比裸 L2 容量小，因为消费者自己的 W/U 写流在同时抢。
- Level 2 的交接是 0.60 GB/调用（`L masked` + `Rk/Rv`）。按 17 MB ring 的回收率，读腿
  0.60 × 1.27～1.43 = **0.76～0.86 ms**。**§11.46 复价的 0.74 ms 是可兑现的，不是上界。**
- 代价是**窗口尺寸被钉死**：C=64 下 `Rk/Rv` 32 KB/chunk、`L masked` 8 KB/chunk，Level 2 的交接约
  40 KB/chunk；17 MB 的 live ring 配 2 槽双 window ⇒ **每窗口 ~210 chunk，全 12288 chunk 约 58 个
  窗口**（只算 `Rk/Rv` 则是 256 chunk/窗口、48 个窗口，脚本打印的是这个口径）。**窗口开大就掉出
  knee，收益按上表衰减——这不是能按 slot 方便随便调的参数。**

  **（§11.48 修正口径：`L masked` 是 fp32 `[64,64]` = 16 KiB/chunk，不是 8 KB，所以交接是 48 KiB/chunk
  而不是 40，knee 处的窗口是 ~175 chunk 而不是 ~210。这一改把 Level 2 从"勉强可行"推到与占用地板
  （一波 ≥ 384 chunk）不相容，判决见 §11.48 第 2 节。）**
- 与 §11.46 第 5 节一致：Level 2 **不吃 stage 配平墙**（它同时削 pre_gram 的写与 solve 两半的读），
  所以这 0.76–0.86 ms 不像 cube 侧候选那样要按 §11.38 的 3.7 倍折扣缩水。Level 3 的 W/U（0.4 GB）
  同理值 ~0.5 ms，但它的读侧在 K2，而 K2 只有 30% 屋顶、是延迟墙不是带宽墙，落地要打折。

### 5. 顺序

§11.28/§11.29 把 Level 2/3 排到最后的两条理由现在都不成立：**价错了 7 倍**（§11.46），而"跨 launch
拿不到 L2"这个隐含担忧被本轮**直接证伪**。**Level 2 应当排到已落地项（§11.37 装载合并 −0.178、
§11.39 P 上片 −0.184）之后的第一位**：0.76–0.86 ms，e2e 的 7–8%，比那两轮加起来还大一倍。
准入条件（slot 字节账、credit/free 协议、同步审计 diff、精度 gate、交错 A/B）不变，但**窗口尺寸要
按本轮的 knee 定**。

复现：`KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_ring_survival.py`
（复用 §11.46 的 cube 转录，无新 kernel）。不合格的第一版仪器同样入库
（`kernels/v1/k1_l2_survival_probe.cpp` + `tools/probe_l2_survival.py`），它现在只会打印
`INSTRUMENT INVALID`，留着是为了让守卫和三版读数可复现。

## 11.48. Level 2/3 的设计判决：一条"一波字节 ≥ knee"的不等式同时关掉两级；raw Gram 的 Fixpipe 实测是负载承载的（2026-09-28）

§11.47 把 Level 2 抬到已落地项之后的第一位（0.76～0.86 ms）。本节按设计书第 365 行的五条准入把它
设计到可实现的粒度，结果**三条独立约束同时不满足**，Level 2 与 Level 3 一起关闭。除第 5 节那一次
ablation 外，本节所有数字都来自已入库的实测。

### 1. 统一机制：跨 launch 之所以贵 7 倍，是因为没有 pacing

launch **内**的交接（`Ga/Gk/Gb` 604 MB、`Aqk32 raw` 与 `L raw` 各 201 MB）按 §11.12 的 0.2 ms/GB
计价，是因为生产者被**片上有界队列**卡住：`kernels/v1/k1_pre_gram_mix.cpp:1106` 的
`CrossCoreSetFlag<2, PIPE_MTE3>(FL_READY)` 与 `:381` 的 `CrossCoreWaitFlag(FL_READY)` 每 chunk-step
握一次手，AIC 侧 `TQue<QuePosition::B1, 4> qa, qb`（`:349`）只有 4 个 L1 槽。所以在途集合是
"每块几个 chunk × 24 块" ≈ 2～3 MB，天然在 L2 里。

跨 launch 的交接没有这个握手：生产者跑到底，消费者才开始，在途集合 = 整个 buffer（201～403 MB），
读价就是 §11.46 的 1.427 ms/GB。**所以要把跨 launch 的读变热，唯一办法是把 pacing 重新装回去，
而装法只有两种：按窗口切 launch，或设备侧 credit。下面两级各证明一种不可行。**

不用新测的佐证：pu 扫描 8/16/32/64（3.518/3.374/3.315/3.271 ms）里，pu=8 的每块在途是 16 chunk
（0.8 MB/块，24 块 ≈ 19 MB）而 pu=64 是 128 chunk（6.3 MB/块，151 MB），局部性差 8 倍，时间却单调
变好——说明 pre_gram 的交接**本来就不受块跨度影响**，即 pacing 是每 step 的、不是每块的。

### 2. Level 2：一波字节 ≥ knee，差 2.2～4.4 倍，无解

先修口径：§11.47 第 4 节写的"`L masked` 8 KB/chunk、合计 40 KB/chunk、每窗口 ~210 chunk"低估了 L。
`L` 是 fp32 `[c_solve, CHUNK, CHUNK]`（`python/kda_ascendc_v1/api.py:659`），16 KiB/chunk；`Rk`/`Rv`
各 16 KiB ⇒ **48 KiB/chunk**（×12288 = 604 MB，与 `docs/artifacts/stage_traffic.txt` 的 0.60 GB 对账
一致）。knee（16.8 MB）处的窗口是 **175 chunk**，不是 210。

| 约束 | 要求 | 来源 |
|---|---|---|
| L2 knee | `lead × W × 48 KiB ≤ 16.8 MB`；depth 2 ⇒ **W ≤ 175** | §11.47 |
| 占用地板 | 一波 = 24 个 MIX 块 × `2pu` chunk ⇒ **W ≥ 48pu**；pu ≥ 8（再小则每块 prologue 吃掉 0.247 ms）⇒ **W ≥ 384** | `api.py:706` 的 pu 扫描 |
| 配平墙 | stage = max(AIV, AIC)，AIC 2.141 / AIV 2.134 ⇒ 收益 = **min(Δ_AIV, Δ_AIC) + 0.007**，不是两者之和 | §11.39 |

**knee 要 W ≤ 175，占用要 W ≥ 384（pu=8）或 768（pu=16）。差 2.2～4.4 倍。** §11.47 表里 89～100%
那一行在 Level 2 上不可达，天花板是 33～67 MB 那个 ~29% 的平台。

唯一"最不难看"的点是 `W=768, pu=16, n=16 窗口`（占用刚好一波、pu 代价最小、ring = 2×768×48 KiB =
75.5 MB ⇒ 按 §11.47 的 67.1 MB 行插值回收 ~27%）：

| 项 | ms | 依据 |
|---|---:|---|
| Δ_AIV = 0.201 GB × 1.227 × 0.27 | −0.067 | 冷价 1.427 减 L2 价 0.2 |
| Δ_AIC = 0.403 GB × 1.227 × 0.27 | −0.134 | 同上 |
| stage 收益 = min + 0.007 | **−0.073** | §11.39 配平墙 |
| pu 64→16 | +0.103 | 实测 3.374 − 3.271 |
| pre_gram launch 1→16（+15 × 4.4 µs） | +0.066 | §11.46 的每 launch 价 |
| solve slice 24→16 | +0.125 | 实测 2.650 − 2.525 |
| **净** | **+0.22（变慢）** | |

即使把 launch 与 slice 两项白送，只剩 `0.073 − 0.103 = −0.03 ms`，最好情况是打平。

补一条反证：窗口化**不会**额外带来 PG‖solve 的重叠收益。§11.27 已量 `PG‖solve:AIC` 只隐藏 0.17、
`PG‖solve:AIV` 只隐藏 0.22——这台机器吞吐饱和（9.27 GB / 10.40 ms = 屋顶的 77%），并发只是交错，
不是相加。

### 3. Level 3：生产者地板很低，但消费者装不上窗口，credit 会死锁

生产者侧确实可行，这是上一轮把 Level 3 排在 Level 2 之后的理由：cube 每块 `WU_NCHUNK=2` chunk
（`api.py:112`），24 块只要 48 chunk，depth-2 在途 = 2×48×32 KiB = **3.1 MB**，远在 knee 之下。
**消费者侧不成立**，三条路都堵：

1. **不能按 launch 切窗口**：`kda_k2_persistent_loop` 是单个 MIX launch，`nblk=24`、每块持 `MAXH=4`
   个 head、**fp32 state 常驻 UB 走完 128 个 chunk**（`api.py:841` 的注释）。按 chunk 窗口切就等于每
   个窗口都要把 state 落 GM 再读回，那是 `SPLIT_STATE_OUT` 的 384 MiB 快照形态（`api.py:872`）：每
   窗口 6.3 MB × 32 窗口 ≈ 200 MB 新增流量，直接吃掉 0.47 ms 的收益。
2. **不能设备侧 credit 自旋**：`api.py:861` 写死 `aic_cores = 24`，而 `nblk = (bh + maxh - 1)//maxh
   = 24`（bh=96、maxh=4）。**K2 的 24 个 MIX 块刚好占满全部 24 个 AIC 并在整个 launch 期间驻留。**
   于是谁先 launch 都死锁：K2 先 ⇒ cube 的窗口块一块也派不进来 ⇒ K2 自旋等 credit；cube 先 ⇒ 它跑完
   窗口 0、块退役，K2 的 24 块立刻占满 AIC 并自旋等窗口 1，cube 剩下的窗口块永远排在 K2 后面。
   Ascend 没有 yield/preempt，自旋没有逃生门（而 §11.34/§11.41 已经记录过两次挂核的代价）。
3. **融合（把 cube 的 W/U 生产并进 K2）更差**：solve stage 从 2.322 降到 max(AIV 2.134, ASM 0.513) =
   2.134，K2 的 AIC 从 4.06 涨到 5.57（+cube 的 1.511），合计 **7.70 对今天 6.32**；即使把省下的 W/U
   往返（0.403 GB 读 × 1.227 + ~0.16 写 ≈ 0.65 ms）全给回来也是 7.05，**净亏 0.7 ms**。机制与 §4.4
   判死融合宽 RHS solve 相同：把活从"被 solve 的 AIV 松弛量遮住"搬到"最长那一段的临界 pipe 上"。

K2 侧的总量也说明为什么没有便宜出路：它的跨 launch 输入是 **912 MB**（W/U 402.65 + Qg/Kg 402.65 +
Aqk16 100.66 + Decay 6.29），一波就把这 912 MB 全部张开着读——**是 16.8 MB knee 的 54 倍**。

**这台机器上"能填满机器的 stage"与"能装进可用 L2 的在途集合"是互斥的**：一波的最小字节数由
`24 块 × 每块最小高效跨度` 决定，而可用 L2 只有 16～32 MB。Level 2 与 Level 3 只是这条互斥的两个
实例，不是两个可以分别攻克的工程问题。

### 4. 实测：raw Gram 的 Fixpipe 是负载承载的（§11.28 那条候选关闭）

两级窗口化都关，剩下的唯一正向候选是 §11.28 表里那条"`Aqk32` raw 的一次复读（201.33 MB），
+0.04～0.10 ms，要动 pre_gram 的 AIV band 循环"。先量它的 store 半边，再决定要不要动那个循环。

改动（形态与 `debugStores`/`a16Mode`/`cube_a16_resident` 一致，生产默认值不变）：
`k1_pre_gram_mix.cpp` 的 `run_gram_aic` 新增运行期参数 `rawMode`（bit0 去 `Aqk32` 的 Fixpipe、bit1 去
`L` 的），入口 `kda_pre_gram_mix` 尾部加同名参数，`api.pre_raw_mode()`（`KDA_PRE_RAW_MODE`，每次调用
读一次）。四处 Fixpipe 各加一个 `if`（`:558-559` 主体、`:581-586` XBAND）。**`Mmad`/`M_FIX` 配对与
末尾的 `CrossCoreSetFlag(FL_DONE)` 一律不动**，所以 `post_gram` 的 `CrossCoreWaitFlag` 仍有对手，不会
挂核。丢弃臂按构造是错的（AIV 照读两个槽），只用它的钟——与 §11.35 的 cube 臂 2、§4.7 的
`a16Mode=2` 同一口径。

仪器 `tools/probe_pre_gram_rawmode.py`，同进程交错 MIN，三次独立运行（5 / 6 / 6 轮；run3 是第 6 节
那次参数顺序修正之后的代码，用来证明结论不依赖参数位置），`[1,8192,96,128]`、C=64、device 3：

| 臂 | 丢弃 store | pre_gram Δ（run1 / run2 / run3） | e2e Δ（run1 / run2 / run3） |
|---|---:|---:|---:|
| 0 shipped | — | 0（4.290 / 4.303 / 4.289 ms） | 0（10.911 / 10.909 / 10.887 ms） |
| 1 去 `Aqk32` raw | 201.33 MB | **+0.022 / +0.007 / +0.013** | +0.036 / −0.008 / −0.006 |
| 2 去 `L` raw | 201.33 MB | **+0.049 / +0.036 / +0.039** | +0.014 / +0.034 / +0.043 |
| 3 两个都去 | 402.65 MB | **+0.085 / +0.063 / +0.085** | +0.059 / +0.064 / +0.052 |

**每一次丢弃都让 stage 变慢，且随丢弃字节单调。** 三次运行符号一致、量级一致（`L` 那一路最稳定，
+0.036～+0.049；`Aqk32` 那一路最小，+0.007～+0.022）。删 store
不可能花带宽，所以这条 ablation **不是一次干净的字节删除**：两个 Fixpipe 同时是**节奏点**。机制：去掉
Fixpipe 后 `SetFlag/WaitFlag<FIX_M>` 在没有 fixpipe 在途时立刻退役，`qc.FreeTensor(cf0/cf1)` 提前落地，
下一个 `Mmad` 就撞上原先被 fixpipe 延迟遮住的 L0C WAR；而 AIC 与 AIV 是每 step 握手的
（`FL_READY`/`FL_DONE`），AIC 被卡住就把整段拖住。这与 §11.44 重验出的"pre_gram 的 MTE3_V marker
仍然负载承载"是同一形状的发现。run2/run3 的可加性守卫都报了 SUSPECT（0.063 与 0.085 对 1+2 之和
0.043 / 0.052），说明两个槽之间还有交互、per-tile 的数不是独立价——但**方向与单调性三次都成立，
判决不依赖可加性**。

判决：**402.65 MB 的 raw Gram store 不是候选，而且它不免费，它是负载承载的。** §11.28 的"`Aqk32` raw
复读 +0.04～0.10 ms"从 store 侧关闭：不要按字节账去把 Fixpipe 拆象限，也不要据此把 mask/scale 搬到
Cube。read 半边这条 ablation 量不到（四臂里 AIV 都照读），而它是 launch 内的，按 0.2 ms/GB 上界也只有
≤0.08 ms，且要动 band 循环——**不做**。

口径提醒：`KDA_PROFILE=1` 的 per-stage 数被它自己每个 stage 前后的 `torch.npu.synchronize()` 抬高了
——本轮 pre_gram 读到 4.30、solve 读到 6.15，而账本的隔离值是 3.31 / 2.32。判决只用同 harness 的
mode 间差值，以及不带 profile 的 e2e；e2e 的 +0.064 与 stage 的 +0.063 吻合，说明这段确实在关键路径
上 1:1 传导。

### 5. 这一轮之后还剩什么

| 方向 | 值 | 状态 |
|---|---:|---|
| Level 2 window ring | 唯一可行点净 **+0.22 ms（变慢）** | **CLOSED**（knee vs 一波字节，差 2.2～4.4×） |
| Level 3 window ring | 上界 0.47 ms，无法装窗口 | **CLOSED**（K2 state 常驻 UB + `nblk == aic_cores` 死锁） |
| cube → K2 融合 | −0.7 ms | **CLOSED**（把活搬到最长段的临界 pipe） |
| raw Gram store | **+0.063～+0.085 ms（负收益）** | **CLOSED**（负载承载，本节实测） |
| 跨 launch 字节总计 | 1.52 GB，其中 K2 侧 0.91 GB | 全部无法 pacing |

**优化程序到此收敛。** 10.40 ms 的端到端里 9.27 GB 已经跑到屋顶的 77%；剩下的字节要么结构上必需
（输入 0.81 GB、launch 内交接 2.0 GB 已按 0.2 计价），要么跨 launch 且无法 pacing（1.52 GB）。调度侧
6 个排水变体在噪声里（§11.45），stage 两半配平到 0.007 ms（§11.39），host 有 ≥5 ms 松弛（§11.28）。
**剩余可动空间是 0.05～0.15 ms 量级的单项，且每一条都已量过或判死。** 要再拿一个 0.5 ms 以上的量级，
只能改变问题本身：state dtype、公式、或 chunk 内并行的算法形态——而设计书 §4.1～§4.5 的四条路线都已
实测判死，重开条件写在 §4.6。

复现：`KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_pre_gram_rawmode.py`
（`KDA_RAWMODE_ROUNDS` 控轮数）。生产默认 `KDA_PRE_RAW_MODE=0`，本轮**没有改变任何生产行为**——
`rawMode` 的 1/2/3 只由探针使用。

### 6. 验证，以及一条参数顺序的契约

`rawMode=0` 时两个 `if` 都为真、四条 Fixpipe 原样执行，所以生产路径按构造位一致。仍然跑了：
`tests/test_dead_store.py` + `tests/test_pre_gram_ub_budget.py`（C=64，6 passed），
`tests/test_kda_bt16.py` + `tests/test_torch_reference.py`（C=64，12 passed / 6 skipped，对 torch
参考实现），`tests/test_kda_bt16.py` + `tests/test_dead_store.py`（C=16，3 passed）。C=32 未重跑。

**第一次跑挂在 `test_dead_store.py::test_the_flag_reaches_both_kernels`**，值得记下来：那个测试的
`_LaunchSpy.last_int()` 读的是 `args[-1]`，契约写在它自己的 docstring 里——"`debugStores` 是尾随的
int32"。把 `rawMode` 加在它后面就把这个契约破坏了，测试报"escape hatch is dead"（拿到 0 而不是 1）。
修法是**把探针参数插到生产 flag 前面**，而不是改测试：`kda_pre_gram_mix` 的签名现在是
`(..., xRowBytes, gRowBytes, rawMode, debugStores)`，`api.py:749` 的参数表同序，kernel 里留了注释说明
为什么 `debugStores` 必须留在最后。

**规则：给生产 kernel 加运行期探针参数时，先查有没有测试按位置读参数表。** 这个仓库里有
（`args[-1]`），而且它是对的——按位置断言才能抓住"flag 没接到 kernel"这类静默失效。

## 11.49. 指令数不是墙：每块 MTE1 指令砍掉 46%、stall 清零，块墙纹丝不动（2026-09-29）

问题（用户侧）：能不能"缩减指令循环、加快指令"来动 assemble 的块墙（§11.45 之后的追问）。

两个新 arm（`k1_solve_assemble_pipe_probe.cpp`，12288 chunk / grid 3072 / 同进程交错 MIN of 5）：

| arm | A16 | ms | vs control |
|---|---|---:|---:|
| mode 0 control | — | 0.390 | +0.000 |
| **mode 7 合并 lift**（repeat 2 + srcStride 2，目的分形连续） | **IDENTICAL** 0/50331648 | 0.388 | −0.002 |
| mode 8 同一置换、另一种旋钮（repeat 2 + dstGap 1） | **DIFFERS** 12558462 | 0.388 | 速度无意义 |

- 每块 MTE1 指令从 52（pass0 4x`LoadData`+1x`LoadDataWithTranspose` ×4 chunk，pass1 4+4 ×4）
  降到 28（−46%）；全块搬运/计算指令 ~110 → ~86。四个单分形调用拼出的置换等价于
  `srcStride=2` + 连续目的（`dstGap=0`）——这不是猜的：pass0 的 B 装载本来就是
  `repeatTimes=KF*KF` 的一次调用。
- **位判据先于计时**：mode 7 位一致；mode 8 判死——`dstGap` 在 `LoadData2DParams` 上的语义与
  "连续目的"的直觉不一致，参数化重排必须逐位验证（`LoadData2dTransposeParams` 的
  `dstFracGap` 是第三个旋钮，没有被这一步用到）。

上板核对（`/data/models/Qwen3-4B/kda_msprof_20260928_rolling/mode7/`，与 mode 0 同批采集）：

| us/块 | control | mode 7 | Δ |
|---|---:|---:|---:|
| aic_time（块墙） | 3.582 | 3.585 | +0.003 |
| scalar | 0.550 | 0.548 | −0.002 |
| **scalar_mte1_stall** | **0.254** | **0.000** | **−0.254** |
| mte1 | 0.162 | 0.148 | −0.013 |
| cube | 0.098 | 0.092 | −0.005 |
| fixpipe | 1.816 | 1.818 | +0.002 |
| mte2 | 1.205 | 1.204 | −0.001 |

判读：

- **7% 的墙（`scalar_mte1_stall` 0.254 us）归零，块墙不动**——这部分本来就压在 MTE2/FIX 的
  数据通路底下（管线本来就有 ~6% 的重叠，见 §11.45 的 `五 pipe 和 / 块墙 = 1.06`）。**指令条数、
  循环、stall 都不是块墙的来源**；这是第 7 个"改结构不动墙"的臂。
- 块墙由两条数据路径决定：fixpipe 1.812 + mte2 1.205 = 3.02 us = **84%**。动墙只有两条路：
  **每条指令搬更多字节**（合并装载已落地，e2e −0.178；上限是整 pass 一次调用的形态，隔离
  0.157 vs 0.217，代价是操作数重排，§11.37 记录在案）与**少搬字节**（P 上片已落地，e2e −0.184）。
- 硬件侧没有"加快指令"的旋钮：频率 1800 = rated 1800、`aic_icache_miss_rate` 0.000、
  `KERNEL_TYPE_AIC_ONLY` 已设。能做的只剩"少发指令、发大指令"，而"少发"这一半已实测收益为零。
- **判决：mode 7 不入生产**（位一致但零收益；RTC 编译期守卫 `mode == 7 && KF == 2`，mode 8 同）。
  继续动 assemble 只剩"整块/整 pass 一次装载（需重排布局）"与 store 形态两条线，都先过 stage/e2e。

复现：`KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_solve_assemble_pipe.py`；
上板：`KDA_MSOPP_KERNEL=kda_solve_assemble_pipe_probe KDA_MSOPP_MODE=7 msprof op ...`
（`tools/msop/run_msop.py` 的 `GEOMETRY` + `KDA_MSOPP_MODE`，归档 INDEX.md 同目录）。

## 11.50. 发大指令判定：整窗装载把块墙砍掉 22%、MTE2 砍掉 64%、隔离 ASM −0.066，但 stage/e2e 仍然不收（2026-09-29）

问题（用户侧）："发大指令这条路线是否可行"（§11.49 留下的两条线之一）。

**形态**：assemble 每块要读 6 类 ND2NZ（pass0 的 Lneg 块 + 4 个 Xb 带，pass1 的 Xb 块 + 4 个 P 带）。
Xb 的块内跨度 `[c0*2*MM, (c0+nch)*2*MM)` 在 GM 里连续，且每个 chunk 的 4 个 16 行带是等距的
（512 元素），所以**一次 `Nd2NzParams(4*nch, 16, M, BANDE, M, 16, 1, BANDE)` 就够**：band0 是 pass0 的
B，band1 是 pass1 的 A——pass1 连装载都不要了，块内 MTE2 调用 `1 + nch + 2` → **2**。
两个附带形态变化，都是位一致的来源：pass1 的 A 在整窗序里是 16 行（n-block major）序，**一次直读**
（`LoadData repeat=KF*KF`）取代原来 4 次交叉调用；pass0 的 B 与 §11.37 的带合并序完全同构
（`LoadDataWithTranspose repeat=KF*KF` 论证相同），所以 L0B 一个字不用改。

第一步在探针里做（`k1_solve_assemble_pipe_probe.cpp` mode 9，先位后时）：
A16 `IDENTICAL 0/50331648`；隔离 0.394 → **0.351 ms**（−0.043 / −11%，复跑 0.398 → 0.353 确认）；
上板 `/data/models/Qwen3-4B/kda_msprof_20260928_rolling/mode9/` 块墙 3.582 → **2.812 us**、
mte2 1.205 → **0.436**（−64%），fixpipe/mte1/cube/scalar 全在 ±0.01 内。

第二步落生产：`k1_solve_assemble.cpp` 加 `loadMode >= 4`（= mode 2 结构 + 整窗装载；P 仍上片、
store 仍逐块；参数表仍是 `(..., C, loadMode)` 两个 int）。**唯一的行为耦合点**：§11.42 的批量 store
从 `loadMode >= 3` 改成 `loadMode == 3`——两个旋钮是两条测过的线（批量 store 隔离赢 stage 亏），
不叠乘。首臂位判据：mode 4 对 mode 0 **0/100663296 elements differ**，state 全同。

**判决（`tools/probe_solve_assemble_loads.py`，12288 chunk / grid 3072，5 臂同进程）**：

| 读数 | mode 2（现生产） | mode 4（整窗） | Δ |
|---|---:|---:|---:|
| 隔离 ASM（MIN of 5） | 0.540 ms | **0.474** | −0.066 |
| 隔离 AIC 半边 | 2.166 | **2.016** | −0.150 |
| 隔离 sliced（生产调度重放） | 2.330 | 2.323 | −0.007 |
| **stage solve（KDA_PROFILE）** | **5.957** | **5.957** | **±0.000** |
| e2e（do_bench median of 3） | 10.417 | 10.418 | +0.001 |

stage 的"平"不是单点：另跑 5 轮交错的同进程复测（mode 2: min 5.970 / med 6.074；
mode 4: min 6.004 / med 6.085）也不收，**且在噪声内还略慢**。
生产 kernel 本体的上板对拍（本目录 `mode2/` vs `mode4/`，同批采集）解释了两件事同时为真：

| us/块 | mode 2 | mode 4 |
|---|---:|---:|
| aic_time（块墙） | 3.7475 | **2.9236**（−22%） |
| mte2 | 1.2031 | **0.4342**（−64%） |
| fixpipe | 1.8131 | 1.8191 |
| scalar | 0.7392 | 0.7035 |
| mte1 / cube | 0.1617 / 0.0993 | 0.1678 / 0.1256 |
| sum/wall | 1.072 | 1.112 |

**判读**：块墙确实降了 0.82 us/块（−22%），但端到端一分不拿。理由是这半边的绝对高度：
整窗之后的 AIC 半边 2.016 已经**低于 AIV 半边 2.136**，而 stage 的 24 个 slice 是
`wide(sa) → event → assemble+cube(sb)` 的两流重叠，关键路径在 AIV/wide 那一半；assemble 的
局部排水（含这次砍掉的 0.77 us/块 MTE2）本来就压在它的 slice 里被吸收（sliced 列 2.330 → 2.323，
−0.007 与 stage 的 ±0.000 同量级）。§11.42 的批量 store 是同一签名（隔离 −0.10、stage 不收），
**这是第二次**：在这个 stage 里，"assemble 的局部时间"已经不是端到端变量。

**判决**：mode 4 作为命名臂保留（默认仍是 mode 2），它是"哪天 assemble 半边被 wide/AIV 提速
暴露出来"时的第一个开关；而"发大指令"这条线就此完整：**少发**零收益（§11.49 实测），
**发大**有实打实的块级收益（−22% 块墙 / −64% MTE2 / 位一致），但 stage 不收。继续压端到端
只剩动 wide/AIV 半边或改切分/重叠结构，assemble 侧到此为止。

复现：`KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_solve_assemble_loads.py`
（含 5 臂位判据 + e2e + 隔离重放 + KDA_PROFILE stage）；上板
`KDA_MSOPP_KERNEL=kda_solve_assemble KDA_MSOPP_MODE=4 msprof op ...`（归档
`/data/models/Qwen3-4B/kda_msprof_20260929_window/`，含 mode2/mode4 对拍与 SUMMARY.txt）。

## 11.51. AIV 半边第一次上板：vec 68%、mte3 17%、mte2 7%——"大指令"在 AIV 上是什么，还剩哪些（2026-09-29）

问题（用户侧）：mode 4 落地后 AIC 2.016 < AIV 2.136，stage 第一次是 AIV 绑；"AIV/wide 那边有没有
能用大指令的方法"。

§11.33/§11.34 的 AIV 账（2.144 = 1.169 递归 + ~0.95 DMA）全部来自重放/消融，从没有上板。本轮补上：
`tools/msop/run_msop.py` 加了 wide 几何（六指针 + `(C, a16Mode, debugStores)`，网格按
`NCHUNK/SUBB`），`tools/parse_msop_pipeutilization.py` 改成自动识别 `aiv_*` 列
（`kda_solve_wu_wide` 是 `AIV_ONLY`，AIC 列全 NA——09-24 那次 assemble 采集的镜像）。

**上板（wide，`/data/models/Qwen3-4B/kda_msprof_20260929_aiv/wide_mode0/`，3072 block）**：

| us/块 | mean | 占比 |
|---|---:|---:|
| aiv_time（墙） | 19.87 | — |
| **vec（递归）** | **13.55** | **68%** |
| scalar | 4.56 | 23% |
| mte3（导出） | 3.32 | 17% |
| mte2（L/L21 gather） | 1.42 | 7% |
| 四 pipe 和/墙 | — | 1.15 |
| scalar_mte3_stall | 2.85 | — |

（profiling 重放的绝对墙比 launch 重放导出的每块值高约 1.4 倍——AIV 核数口径未在工具里固定，所以
**只用比值与同条件差值**，不用绝对每块。AIC 的 assemble 采集两种口径只差 6%，这里记一笔。）

**消融在同批采集里的重测**（mode 1 = 不写 strict-upper blank，故意不正确）：墙 19.87 → 19.30
（**−0.58 us/块，−2.9%**），mte3 3.32 → 2.77、scalar_mte3_stall 2.85 → 2.31，**vec 不动**
（13.55 → 13.51）。blank 就是暴露在 mte3/scalar 上的那 0.58。

**关键：AIV 第一次是 AIC 的下界，AIV 削减的符号翻转了**（`tools/probe_solve_a16_ablation.py`，
`KDA_ASM_LOADS=4`，同进程三臂）：

| 口径 | mode 0 | mode 1 去 blank | mode 2 去整块 |
|---|---:|---:|---:|
| e2e（旧世界，§11.34，AIC 绑） | 10.623 | +0.041 | +0.224 |
| **e2e（本世界，mode 4，AIV 绑）** | 10.395 | **−0.077** | **−0.057** |
| sliced 重放（旧） | 2.535 | −0.002 | +0.085 |
| **sliced 重放（本世界）** | 2.328 | **2.239（−0.089）** | **2.242（−0.086）** |
| AIV 重放 | 2.126 | 2.039 | 1.851 |
| AIC 重放 | 2.024 | 2.030 | 2.020 |

**判读**：stage ≈ max(AIV, AIC) + ~0.2（24 slice 的事件/间隙）。AIV 削 0.087 → sliced −0.089（1:1）；
AIV 削 0.275 → sliced 只 −0.086，因为削穿了 AIC 的 2.02 线，AIC 接管（max 模型三臂全对得上）。
⇒ **AIV 侧的"可兑现头寸" ≈ 0.10 ms**（把 2.126 削到 2.02 以下为止是 1:1；再往下必须成对削 AIC）。

**"大指令"在 AIV 上是什么、还剩什么**：

1. **AIV 的大指令轴是 repeat 宽度（chunk 实例数）**，它已经拉满：`MulAddDst` 一条指令 = NC=8 个
   (sub-block, chunk) 实例 × 32 lane；递归的指令数 per chunk = `SB·M(M+3)/2 / NCH`（当前 140）。
   翻倍到 NC=16 需要 UB 244 KB > 192 KB（现在是 134 KB：lraw 32 + af 32 + ab 16 + cexp 8 + L21 24
   + 零/eye 10）——**UB 是天花板，不是描述符**。而算法上更宽的融合形态（RHS 256 lane）§11.33
   实测 2.5–5× 慢。vec 13.55 = 68% 的墙说明：**AIV 的墙是递归的"工作量 × 指令数"，不是"指令太碎"。**
2. **DMA 侧已经不小**：gather 每次 512 B（4 chunk × 128 B）、导出每次 2 KB，且合并被三件事封死——
   Brcb/repeat-stride 要求 tile 是 `[row][chunk][lane]` 且块内 8 列连续；GM 是 `[chunk][row][lane]`；
   `DataCopyParams` 只有 2D 且 gap 无符号（s=0/s=1 之间是反向跳）。mte2 只占 7%：**就算合并，
   上限也是小头**。
3. **能动的三类**（按性价比）：
   - **把 export 挪给 AIC**：blank 已在上面定价（AIV −0.58 us/块）；正确版本 = assemble 用一次
     零 Mmad 的 L0C + 4 次 fixpipe 把 strict-upper 写掉（AIC 目前有 ~0.10 ms 松弛），不是"删掉"
     （Cube 必须读到 0，§11.34 的教训）。
   - **指令节食**：生产路径里 `debugStores=0` 时 wide 的 `Cast(af←ab)` 回转是 dead code（fp32 af
     只有 A32 debug store 用）——CH=8192 元素 × 一次 cast/块，纯省。
   - **结构性（真正的大头）**：**SB=4 / M=16**。递归指令数 per (sub-block, chunk) 实例 ≈ M²/2：
     M 减半让每实例指令数 /4，每 chunk 的实例数 ×2 ⇒ per chunk 指令数 **/2**，而 UB per 实例 ∝ M²
     也 /4（134 KB → ~34 KB，NC 可以再翻倍到 16 ⇒ 再 /2）。代价是耦合块全部变多：
     SB=2 每 chunk 2 个 32³ Mmad，SB=4 是 12 个 16³（MAC 数相当 65.5k vs 49.2k）——**把 AIV 的
     递归搬给 AIC 的 Cube**，而 AIC 现在有松弛。这条要自己的探针，是本轮之后唯一能过半的 AIV 侧路线。

归档：`/data/models/Qwen3-4B/kda_msprof_20260929_aiv/`（wide_mode0/1 + INDEX.md + SUMMARY.txt）；
采集命令 `KDA_MSOPP_KERNEL=kda_solve_wu_wide KDA_MSOPP_MODE=<a16Mode> msprof op ...`。

## 11.52. AIV 侧的"大指令"旋钮：NCHUNK 8→12 位一致但只值 −0.03 ms；上限是 UB，不是描述符；SB=4 的账（2026-09-29）

§11.51 把 AIV 的墙定在 vec（68%）之后，问题变成"每条指令覆盖的实例数还能不能加"。宽的两种写法
（NC↑ / 融合 RHS）一个被 UB 卡住、一个 §11.33 判死，剩下第三条是**每块覆盖更多 chunk**：
递归的指令数 per tile = `M(M+3)/2`（32 Adds + 32 Brcb + 496 MulAddDst），与 tile 里有多少实例无关，
所以 per chunk 指令数 = `M(M+3)/2 / NCH`，NCH = NCHUNK/SB —— **140（NCHUNK 8）→ 93（12）→ 70（16）**，
而且**耦合形态一字不动**（仍是每 chunk 一个 off-diagonal 块、ASM_NCHUNK 4），这是它比 SB=4 便宜的
地方。约束只有 UB：tile 随 NCHUNK 线性涨（NCHUNK 8 时 134 KB / 192 KB）。

**独立计时**（`tools/probe_solve_wide_nchunk.py`，12288 chunk、真值 L、先位后时）：

| NCHUNK | NCH | grid | wide 单发 MIN of 5 | 位判据（A16/Xb/Lneg 摘要） |
|---:|---:|---:|---:|---|
| 8 | 4 | 3072 | 1.566 ms（1.566–1.590） | 基线 |
| **12** | 6 | 2048 | **1.437 ms（1.437–1.480，−8.2%）** | **与 8 逐位相同** |
| 16 | 8 | 1536 | — | **vector core exception（UB 装不下，响亮失败）** |

**全流水 A/B**（`/tmp/wide_e2e.py`，同协议，两进程 × 两种 ASM 模式）：

| NCHUNK / ASM | e2e 中位（3 轮） | solve_ms | AIV 重放 | AIC 重放 | sliced |
|---|---|---:|---:|---:|---:|
| 8 / 2 | 10.414 | 6.100 | 2.121 | 2.167 | 2.333 |
| **12 / 2** | **10.386（−0.028）** | 6.305 | **2.084** | 2.170 | **2.299** |
| 8 / 4 | 10.397 | 6.230 | 2.139 | 2.028 | 2.319 |
| **12 / 4** | **10.388（−0.009）** | 6.266 | **2.085** | 2.024 | 2.313 |

全流水输出在四种组合下**全部逐位相同**（两次独立的跨进程 digest 都 IDENTICAL：assemble 的 mode
0/2/4 × wide 的 8/12 混搭与 `KDA_WIDE_REF` 比对）。但 e2e 只有 −0.03/−0.01 ms 量级：整网格单发
−8.2% 是真的，流水里 24 个 slice 把它摊薄成 −0.04（AIV 重放），并且 `solve_ms` 在进程间有 ±0.13 的
漂移，不能用它做单点比较（同臂三轮 6.227/6.230/6.270 与异臂 6.100 重叠）。

上板复测（`/data/models/Qwen3-4B/kda_msprof_20260929_aiv/wide_mode0_n12/`，2048 block × 6 chunk）按 **per chunk**
归一后：墙 4.97 → **4.63**、vec 3.39 → **2.78**、scalar 1.14 → 0.77（都是指令数那一路在降），但
**mte3 0.83 → 1.15、mte2 0.36 → 0.42 反向涨**——export/blank 是 per-chunk 工作量，NCHUNK 不动它，
且在更宽的块里反而更贵。**NCHUNK=12 之后的 AIV 墙已经一半不是 vec 了。**

**判决**：NCHUNK=12 **保留为旋钮**（`KDA_SOLVE_WIDE_NCHUNK`），不改默认——收益 0.3% 且部分在噪声里，
和 §11.50 的 mode 4 同一个待遇；16 装不下，这条轴到头。

**SB=4 的账（唯一还能过半的 AIV 路线，先算不动手）**：
- per chunk 指令数 = `SB·M(M+3)/2 / NCHUNK`：SB=4/M=16/NCHUNK=32 → **19 条/chunk（vs 现在 140，7.4×）**；
  UB 同步缩：NC·M² 从 8·1024 到 32·256，lraw+af+ab+cexp+L21 ≈ 110 KB ✓ 装得下。
- 耦合（AIC/Cube 侧）MAC 数**不变**：每 chunk 仍是 `PC³/4`（SB=2：2 个 32³；SB=4：16 个 16³ = 65.5k MAC）；
- 但耦合的**块数 1 → 6、导出字节 2 KB → 3 KB/chunk**，而 assemble 的墙 62% 是 fixpipe（行级 strided
  store，0.46 us/块/2 KB，按行计费）——6 块 × 16 行 = 96 行/chunk vs 现在 32 行/chunk，**推算 AIC 会涨
  到 ~1.4 us/chunk（3×）**，把 AIV 省下来的（vec −2.4 us/chunk 量级）吃掉大半。⇒ SB=4 的成败取决于
  耦合的 **store 形态**（ndNum 批量写 vs 逐块 strided），必须自己的探针；wide kernel 本身不用改
  （blank 循环对任意 SB 都成立），要写的是广义 coupling kernel。

复现：`KDA_CHUNK=64 KDA_SOLVE_WIDE_NCHUNK=12 python3 -u tools/probe_solve_wide_nchunk.py`；
上板 `KDA_MSOPP_KERNEL=kda_solve_wu_wide KDA_MSOPP_MODE=0 KDA_SOLVE_WIDE_NCHUNK=12 msprof op ...`。

## 11.53. AIV 的 export 半边：L21/Lneg 从 64 次调用压到 8 次，逐位一致，e2e −0.10 ms（2026-09-29）

§11.52 把 NCHUNK 拉满后 AIV 的墙一半不是 vec 了（mte3 1.15 + mte2 0.42 + scalar 0.77 per chunk），
本轮按用户点单做 export 半边。**最贵的一条是 Lneg**：旧形态每块 32 次 gather（每行一次，NCH 个
burst）＋32 次 scatter（每行一次，目的跨 chunk 跳 62 block），每次只有 256 B。

**形态改写（纯实现，无新旋钮）**：L21 tile 从 `[row][chunk][lane]` 换成 `[chunk][row][lane]`——
每 chunk 一个连续的 `[M, M]` fp32 块，正是 Lneg 在 GM 里的布局：
- gather：`DataCopyParams(M, M/8, (PC-M)/8, 0)`，每 chunk 一次 2D 拷贝（32 行 × 128 B，源行距 PC）
  = **4 次/块**（4 KB 级），旧的 32 次；
- negate＋cast：同一个 flat 调用（`NCH*MM` 元素，值不变，只是 tile 序变了）；
- store：`DataCopyParams(1, MM/16, 0, 0)`，每 chunk 一次连续 2 KB = **4 次/块**，旧的 32 次。

**位判据先于计时**：wide 单发探针的 A16/Xb/Lneg 前 64 chunk 与旧代码 `torch.equal` 全 True
（262144 + 131072 + 65536 元素）；全流水输出对旧参考也 IDENTICAL（ASM mode 2/4 × NCHUNK 8/12 全过）。

| 口径 | 旧 store | 新 store | Δ |
|---|---:|---:|---:|
| wide 单发 MIN of 5（NCHUNK 8） | 1.566 ms | 1.523 | −0.043（−2.7%） |
| wide 单发 MIN（NCHUNK 12） | 1.437 | **1.406** | −0.031 |
| AIV 重放（NCHUNK 8，ASM4） | 2.139 | **2.037** | −0.102 |
| sliced 重放（同上） | 2.319 | **2.219** | −0.100 |
| e2e 中位（同上） | 10.397 | **10.314** | −0.083 |
| AIV 重放（NCHUNK 12，ASM4） | 2.085 | **1.987** | −0.098 |
| sliced（NCHUNK 12） | 2.313 | **2.205** | −0.108 |
| e2e（NCHUNK 12，ASM4） | 10.388 | **10.293** | −0.095 |
| e2e（NCHUNK 12，**生产默认 ASM2**） | 10.386 | **10.319** | −0.067 |

上板（`…/kda_msprof_20260929_aiv/wide_mode0_n12_l21/`，6 chunk/块，与旧 store 的 n12 同条件）：
mte3 6.90 → **5.42**、`scalar_mte3_stall` 6.35 → **3.49**、vec 16.70 → 16.72（没动）、墙 27.78 → **26.42**；
per chunk：墙 4.63 → 4.40、mte3 1.15 → 0.90。**收益全部来自 mte3 与其 stall**，与改写的目标一致。

**判读**：AIV 的"export 形态"这一刀值 ~0.10 ms（e2e/AIV/sliced 三个口径同向、量级一致，且在
生产默认下也有 −0.067），是 §11.50 之后最大的一次 AIV 侧兑现。**新平衡**：AIV 1.987 已经低于
AIC 2.033（NCHUNK 12）——AIV 的头寸（≈0.10 ms）刚好用尽，下一步要动端到端必须回到 AIC 半边
（assemble 的 fixpipe 1.82/2.92，或 cube 的 A16 重读），或者动 SB=4 那条（§11.52 的账：它同时改
两边的平衡）。store 改写按实现落地（没有旋钮）；NCHUNK=12 仍是旋钮（默认 8，收益 0.02–0.03 且
依赖它自己的编译期几何，先不动默认）。

## 11.54. SB=4 路线执行：耦合核三个根因全修，两边的平衡实测为 AIV −0.43 / AIC +1.67 → e2e +2.05 ms，判亏（2026-09-29）

§11.52 把 SB=4 列为唯一"同时改两边平衡"的路线（AIV 的指令数 140→19/chunk，代价是耦合的块数 1→6、导出 2→3 KB/chunk），并把成败押
在耦合的 store 形态上。本轮把它落地成 `kernels/v1/k1_solve_assemble4.cpp`（六个 16×16 严格下三角耦合、L0C→L1(NZ)→L0B 三层中继、
ndNum 分组 store），先过门再计时，最后同会话判决。

**落地过程修掉三个根因**（每个都有实测签名，位判据先于计时）：

1. *宿主布局*（探针侧）：wide kernel 导出的 bundle 是 `[n10][n32][一个 32×32 行主序矩形(rows 2M..4M, cols 0..2M)]`，不是 4 个独立
   16×16 块。按矩形写之后，靠这片矩形喂的三个耦合块先变好。
2. *同一 L0C 槽连续累加 Mmad 的 C 竞争*（kernel 侧）：Ea/Gr/Gl 三条链原来各是 2/3 个小 Mmad 落在同一 L0C 槽，第二个
   `cmatrixInitVal=false` 的 C 读会抢在第一个的 C 写之前（真机恒错 0.3–0.5，其余 12 项精确；CANN 自带 matmul 的 workaround 是每条小
   Mmad 后无条件 `PipeBarrier<PIPE_M>`，`adv_api/detail/matmul/stage/compute/mmad_compute.h:79`）。修法是**消读写而不是加栅栏**：三项的
   K 堆叠成单次 Mad（K=32/32/48），mad 指令 16→12/chunk，MAC 数不变。
3. *FIX→L1→LoadData 的真实排序*：`PipeBarrier<PIPE_FIX>` 只排 FIX 队列自身，下一个 level 的 LoadData 照样抢在 fixpipe 落 L1 之前（raw
   门错在 (2,0) 5.025e-01 / (3,0) 4.710e-01 / (3,1) 4.557e-01——恰好是所有操作数跨这条边的块）。改用 `PipeBarrier<PIPE_ALL>`，即
   `k1_solve_assemble.cpp:365`（"Pass 1 reads the P tiles pass 0 wrote"）在 mode 2 里被位级验证过的同构形态。`HardEvent::FIX_MTE1` 在整个
   CANN 9.1.0 安装里只存在于 enum（impl 从不使用），token 形态会在第一个等 fixpipe 产物的 level 挂死：C=4/NC=4 在 level 3 aicore
   timeout 且 fixp 置位。

**位判据**：

| 口径 | 结果 |
|---|---|
| raw 门（C=4/NC=4，真事件） | 六块 max\|d\| **2.544e-03**（修前 5.025e-01），strict-upper 精确 0；MIN 0.048 ms（4 chunk，发射开销主导） |
| 全网格（12288 chunk，真 wide 填数，NC=4） | 16 块 **2.737e-03** @ (1,0)，strict-upper 0.000e+00；MIN **1.990 ms**（161.9 ns/chunk，0.648 us/块） |
| 同上 NC=8 | MIN **1.901 ms**（154.7 ns/chunk，−4.5%）：level 屏障不是墙 |
| wide@SB=4（NCHUNK 32，grid 1536） | MIN **1.137 ms**（92.5 ns/chunk）；对角块 9.764e-04、strict-upper 0、Lneg bundle 98304 元素**逐位 True** |

对照 SB=2 的 wide（§11.52 口径：NCHUNK 8 → 1.566 ms、NCHUNK 12 → 1.437 ms）：**−27%（−0.43 ms）**，只有指令数预测（7.4×）的一个零头，
原因见下。

**两边的平衡与 e2e 判决**（同会话两臂；SB=2 臂与 `/tmp/wide_ref8.pt` **逐位 IDENTICAL**，顺带复验了参考文件）：

| 臂 | e2e 中位（3 轮） | solve_ms | AIV 重放 | AIC 重放 | sliced |
|---|---|---|---:|---:|---:|
| SB=2（生产，NCHUNK 8） | 10.316 / 10.327 / 10.334 | 6.559–6.642 | 2.051 | 2.165 | 2.254 |
| SB=4（NCHUNK 32，ASM_NC 8） | 12.368 / 12.381 / 12.382 | 6.718–6.865 | 1.944 | 3.838 | 4.287 |

AIC 拆开：SB=4 = 耦合核 1.90 + cube 1.511 ≈ **3.41**（+重放开销 → 3.838）；SB=2 = assemble 0.513–0.569 + cube 1.511 ≈ 2.02（→ 2.165）。
AIV 重放只省 0.107（隔离口径 0.43），AIC 多 1.67，**e2e +2.05 ms**、sliced +2.03——§11.52 要的"两边同时改平衡"没有发生，瓶颈从 AIV
转到了 AIC。**判决：SB=4 路线亏。**

**为什么救不回来（上板 PipeUtilization，`/data/models/Qwen3-4B/kda_msprof_20260929_sb4/`）**：

* 耦合核每块（4 chunk）墙 12.90 us：**mte2 7.26（56%）**、fixpipe 4.52（35%）、scalar 1.07、mte1 0.43、**cube 只有 0.22（2%）**，
  sum/wall = 1.047（三段基本串行）；`aic_scalar_cube_stall` 7.14 us/块 = 16 条小 mad 逐条 `M_FIX` 的等待；mte2 的 7.26 us 对应
  20 KB/块的行粒 32 B 填充（384 行/块）。Task Duration 1989.3 us 与隔离 1.990 ms 对上。
* 结构上：MAC 数与 SB=2 相同（PC³/4），但粒度细 8×——每 chunk 12 条 mad（9 条单发 + 3 条 K 堆叠，等价 16 个 16³）+ 32 条 LoadData + 9 次 NZ 中继 + 3 次分组 RM store ≈ 56 条
  指令，vs SB=2 的 2 条 32³ mad。把 mte2 布局、NZ 批量化、`M_FIX` 批量化全做满（乐观 −30%）耦合核仍 ≥1.3–1.5 ms，≥2.5× SB=2 的
  assemble，AIC 仍 ≥3.0 > AIV 1.94——**§11.52 押的 "store 形态" 被证伪**：ndNum 分组 store（3 次/chunk）已经落地，墙不在 store。
* AIV 侧同理见顶：SB=4/NCHUNK 32 每 chunk vec 2.78→1.01（−64%，指令数兑现），但 **mte3 0.90→1.90（+111%）**（四个对角 tile + 六段
  bundle 的导出 vs SB=2 的两个 + 一段），AIV 的墙变成 mte3（54%），`aiv_scalar_mte3_stall` 12.96 us/块。

**形态**：`KDA_SOLVE_WIDE_SUBB=4` 保持非默认旋钮（生产仍 SB=2）；kernel 文件、SB=4 探针、上板归档、msop 几何（`run_msop.py` 新增
`kda_solve_assemble4`）全部保留；`tests/test_solve_assemble4.py` pin 住合同（六块 fp64 门、storeMode 0/1 逐位、尾部旋钮槽位）。下一步要
动端到端仍只能回 AIC 半边（§11.53 的结论不变）。

**工具侧同轮修复**：`tools/probe_solve_assemble4.py` 的启动参数补齐 `ab`/`level` 两个尾部槽（旧形态少传两个 blob，会把 chunk 数读成
ablation mask）；`/tmp/wide_e2e.py` 的 rest 集合加 `kda_solve_assemble4` 并加容差比对（SB=4 的 bf16 中间量与 SB=2 非逐位）；SB=4 的
e2e 输出另存 `/tmp/wide_ref_sb4_e2e.pt`，`/tmp/wide_ref8.pt` 由 `wide_ref_old_e2e.pt` 恢复并复验逐位一致。一个诚实记录：本轮 arm A
首跑遇到一次 `kda_pre_gram_mix` 任务超时（507014，16:13:44；同窗口 arm B 正常、重跑 arm A 正常），按偶发记录，不在本轮结论里。

回归：`tests/test_solve_assemble4.py`（新，3 项：六块 fp64 门 / storeMode 0-1 逐位 / 尾部旋钮槽位）通过；C=64 的 `test_solve_assemble_loads.py`、`test_solve_a16_ablation.py`、`test_solve_cube_a16_resident.py` 与 C=16 的 `test_kda_bt16.py`、`test_torch_reference.py` 全过（按设计 skip 的除外）。`test_rtc_compile_config.py` 有两项失败，逐条核对到 HEAD 上同样失败（并行会话新加的探针 `tools/probe_solve_assemble_nc.py` 直接调 `rtc_compile`、若干 `*_probe.cpp` 读 KDA_CHUNK 而不在 api.py 引用表里），与本次改动无关。

## 11.55. pre_gram 的"能不能压缩"结案：指令流就是边际关键路径（加/删双向实测），§11.21 的"减指令"冻结重开（2026-09-30 深夜）

设备 3、KDA_CHUNK=64、`[1,8192,96,128]`、96 块 × (2 AIV subcore, 64 chunk)，MIN of 4、launch 重放；工具
`tools/probe_pre_gram_addwork.py`、探针 `kernels/v1/k1_pg_addwork_probe.cpp`（不在 `api._SOURCES`，生产路径零改动）。

**上板账**（归档 `/data/models/Qwen3-4B/kda_msprof_20260930_pregram/`，msopprof PipeUtilization：
Task 4111.96 us、grid 96 AIC + 192 AIV、1800 MHz）：

* AIV 每 subcore 墙 820.9 us：vec 647.2（78.8%）、mte2 174.7（40.8 GB/s）、mte3 162.9（54.1 GB/s）、scalar 216.7；
  `scalar_vector_stall` 612.3 而 `scalar_mte2/mte3_stall` 全为 0；四项和/墙 = 1.464（有重叠）。
* AIC 每块墙 819.4：cube 28.9（3.5%）、mte1 31.5、mte2 128.5、fixpipe 117.0、scalar 233.9
  （其中 `scalar_mte1_stall` 124.0、`scalar_cube_stall` 36.2）；和/墙 = 0.659。
* 两边平衡（819.4 vs 820.9）；96 块 / 20 核 = 5 个 wave × ~819 us ≈ 4097 + 尾巴 ≈ 4112 us。

**矛盾**：79% vec 忙、零内存停顿 ⇒ "加/减指令 1:1"；但 §11.20 实测每 chunk 删 ~128 条向量指令只换 1.2%，
§11.21 据此冻结了"减指令"路线。

**裁决实验**（同一条 launch 重放，一个二进制两个旋钮）：`addWork` 在每趟 Gram 块之前注入 addWork 条 NG 宽
`Muls(zz, zz, 1.0f)`——`zz` 正是下一条 `Muls` 的输入，链活着丢不掉；乘 1.0 又是 IEEE 恒等，所以
probe@0/@32 全链路 out/state 逐位一致（0/100663296 与 0/1572864）。`ablate` 位掩码按 §11.16 的删块法只读时间：

| 臂 | launch | Δ |
|---|---:|---:|
| control（= 出厂 4.11–4.19 ms） | 4.2 ms | — |
| +32 条/趟 | 5.5 | **+1.3** |
| −gate cumsum（63 条串行 Add/chunk） | 3.8 | −0.3 |
| −sigmoid（6 条/趟 ×4） | 3.8 | −0.3 |
| −cumsum,sigmoid | 3.6 | −0.6（≈可加 ✓） |
| −post_gram（含 FL_DONE 等待） | 3.8 | −0.3 |
| cumsum→朴素 log-scan（6 趟） | 4.1 | −0.1（数值坏：repeat 级 RAW → NaN） |
| cumsum→无冒险分块扫描（35 条 vs 63 条） | 4.0 | **−0.1（数值有效）** |

* **价格：+1.3 ms / 40960 条（32×4×64×5 wave）= 31.7 ns/条（≈57 cyc @1.8 GHz）**，与 §11.16 账本
  （sigmoid 0.204 ms ≈ 24 条/chunk × 64 × 5 × 30 ns = 0.23）对得上——**指令流就是边际关键路径，加与删对称**。
* §11.20 的 1.2% 不是"指令不重要"，而是那次删除落在 FL_DONE 到达上被吸收；§11.21 据此把 cumsum 下修到
  "~1%" 是同一来源的错账。**重开条件：任何"删块后墙不动"的结论必须用本表的加/删双向法重验**（§11.21 表中
  两行已加注）。

**对"pre_gram 有没有压缩空间"的结案**：有，但今天是逐块的 0.1–0.3 ms，没有大头。

1. **cumsum：删块上限 0.3 ms，已实现的无冒险分块扫描兑现 0.1 ms**。朴素 log-scan 数值坏——六趟里后趟
   读到同一条指令自己写的行（repeat 级 RAW，硬件不保序），全链路 100663296/100663296 差、max|d|=nan。
   改写成三相位、写读分离的 radix-8 分块扫描（A：块内 8 行 7 层；B：8 个块总计串行 7 步；C：回加用广播源、
   且跳过块尾行，14 条背靠背），35 条向量指令 + 15 条 barrier 替掉 63+63：launch 4.0 vs 4.2 ms，
   **数值有效（out 944586/100663296 元素不同、max|d| 3.05e-5；state max|d| 9.63e-5——比闸门自身余量
   8.6e-3/5.5e-4 低两个数量级）**，但非逐位。与 0.3 ms 上限的差是 15 层依赖深度 + barrier/repeat 开销，
   不可再压；生产化走容差闸门（非 bit 一致），已落 `kernels/v1/k1_pre_gram_mix.cpp`（与探针同源、无
   ablate 分支）；e2e 判词见 §11.56。
2. **sigmoid：0.3 ms**，但已是 6 条/趟的最小 sigmoid 形态（Muls/Exp/Adds/Dup/Div/Muls），压不动。
3. **post_gram：0.3 ms** = ~0.17 的 FL_DONE 协议等待（两次挂死史，维持冻结）+ 必需的选择/取整/读写。
4. §11.16 其余大项（early stores 0.22、RowReduce 0.19、Exp 0.175）都是承重数学或必须 I/O。

→ 今天可兑现 = **0.1 ms / 4.2 ms launch（e2e 10.4 ms 口径 ≈ 1%）**：cumsum 扫描已验证；sigmoid（0.3 ms）
是 6 条/趟的最小形态、post_gram（0.3 ms）是协议等待 + 必需 I/O，两个"删块上限"目前都没有对应的合法实现；
合删臂 −0.6 ms（−14%）证明系统离 AIC/握手的底还有余量，缺的是代数/协议，不是指令。

复原：`KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_pre_gram_addwork.py`。
诚实记录：首个逐条隔离标定 kernel（已删）用 GetValue/SetValue 收尾，两次把设备打进 "scalar 访问 internal
buffer 越界" 的 aivec 异常（npu-smi 随后正常）；隔离标定没做——本表的在链路价格已由加/删两向互验，不依赖它。

## 11.56. 分块扫描落生产 + e2e 判词：设备腿 −0.115 ms（上下文内 8/8 与 6/6 轮分离），wall 透传 −0.054/−0.096（16 轮配对 14/15 轮为负），删块对照只透传 44–65%（2026-09-30 凌晨）

§11.55 的三相位 radix-8 扫描本轮落进生产 `kernels/v1/k1_pre_gram_mix.cpp`（63 步串行 cumsum → 35 条向量
指令 + 15 条 barrier；与探针同源、无 ablate 分支）。数值对出厂串行基线（非 bit 一致，容差闸门口径）：
out rel 3.47e-04 / max|d| 3.05e-05、state rel 7.60e-05 / max|d| 9.63e-05。三个新工具把"设备侧价格"与
"wall 动不动"分开测：

**① 上下文内设备账**（事件对只包 pre_gram 这一条 launch、交替、MIN of N；`tools/probe_pg_scan_timeline.py`
与 `tools/probe_pg_cumsum_scan_e2e.py` 的 2 事件臂）：

| 进程 | 串行 span | 扫描 span | Δ | 分离度 |
|---|---:|---:|---:|---|
| 时间线运行，8 轮 | 4.137 | 4.023 | **−0.114** | 8/8 轮分离（4.150–4.156 vs 4.035–4.042） |
| 4 臂运行，6 轮 | 配对中位 −0.119 | | **−0.115（MIN 口径）** | 6/6 轮分离（4.141–4.147 vs 4.013–4.036） |

与 §11.55 的隔离 launch 重放（4.2 → 4.0，三次复现）同值：扫描确实把这条 kernel 做快了 ~0.11 ms，
且这个价格在流水里（solve 的 24 片与 k2 都在场）原样存在。

**② e2e wall 账**（同进程、严格交替；`KDA_PG_E2E_EVENTS=0` 去掉每-launch 仪器，跨进程漂移 ±0.15 ms
被逐轮配对差消掉）：

| 运行 | 串行 MIN | 扫描 MIN | MIN 差 | 逐轮配对差（去第 0 轮） |
|---|---:|---:|---:|---|
| 16 轮（无事件） | 10.873 | 10.802 | −0.071 | **中位 −0.054、均值 −0.068，14/15 轮为负** |
| 6 轮（带 2 事件） | 10.968 | 10.853 | −0.115 | 中位 −0.096、均值 −0.092 |
| 8 轮（早先，无事件） | 10.917 | 10.923 | +0.007 | 同进程漂移 0.5 ms，判"不可分辨"（见诚实记录） |

**③ 删块对照**（同进程 6 轮；设备 span 与 wall 的配对差中位）：−cumsum −0.336 / **−0.217（透传 65%）**、
−cumsum,sigmoid −0.556 / **−0.245（44%）**；wall 对 pre_gram 的透传随删除量衰减 —— 流水里有 ~0.2 ms 级别
的松弛，扫描的 −0.115 落在 50–80% 透传带（两次运行 −0.054 / −0.096）。

**④ 端到端账目结构**（`tools/probe_pg_host_floor.py`；宿主 load ~200）：单次调用 wall 10.73–11.0 ms =
主机入队 6.7–7.5 ms（其中 5.2–5.7 ms 是不含 launch 的主机纯工作；74 条 launch ≈ 1.5 ms ≈ 20 us/条）
+ 设备腿 ~10.4 ms（pre_gram 4.05 + solve 24 片 ~2.4（差得）+ k2_persistent_loop 3.99，一条流串行；
74 = pre_gram 1 + solve 3×24 + k2 1）。设备腿 > 主机腿 ⇒ 端到端仍是设备主导，但 wall 透传不是 1:1（见 ③）。

**判词**：
* 设备账实：**−0.115 ms / 4.05 ms pre_gram**，两次独立运行（8/8 与 6/6 轮全分离），与隔离重放同值；
* e2e 账：**−0.054（中位）/ −0.068（均值）/ −0.096（6 轮那次）ms**，对 10.87 ms 是 **−0.5% ~ −0.9%**，
  即 §11.55 的"≈1%"是上限、兑现落在 0.5–0.9%；跨进程直接比生产 wall 永远测不出这个量级
  （漂移 ±0.15 ms），必须用同进程逐轮配对；
* 诚实记录：另一个 8 轮进程里整组臂（含 −0.556 的删除）被压平在 10.96，连 −0.5 ms 的删除都不动 ——
  宿主负载 200、单次调用 wall 跨进程 10.73–11.0 的抖动就是这个量级；所以本节的判词只认配对差，
  不认"生产 wall 前后比"。

复原：`KDA_CHUNK=64 ASCEND_RT_VISIBLE_DEVICES=3 python3 -u tools/probe_pg_cumsum_scan_e2e.py --ref /tmp/pg_scan_ref.pt`
（`KDA_PG_E2E_ARMS` 选臂、`KDA_PG_E2E_EVENTS=0` 去事件、`KDA_PG_E2E_REPS` 轮数）、`... tools/probe_pg_scan_timeline.py`
（全 74 launch 时间线）、`... tools/probe_pg_host_floor.py`（主机地板/launch 计数）。
回归（扫描是数值改动）：`bash tools/run_chunk_matrix.sh` → **C=16 / C=32 / C=64 三档全过**（C=16 是 M=16、
NB=2 的扫描形态；三档都跑 test_chunk_shape_matrix + test_stability_gate + test_c64_gate_overflow）。
