# AscendC 优化代码审查（2026-09-05）

本轮结论：先完成设备端计算闭环，再优化 K2 常驻状态与 Cube/Vector 流水。现有端到端入口主要在 CPU 上运行，不能用它衡量 AscendC kernel 的性能上限。

## 首轮静态审查范围与证据边界

更新：随后已完成服务器密码登录、源码比对、smoke 与 forward 基线复测。最新证据见文末“服务器复核与实测”；以下未登录说明仅描述首轮静态审查时的状态。

- 阅读当前工作树 `src/kda_bt16/kernels.py`、`aclab`，以及 `kda_complete_20260905.tar.gz` 内的原型源码。
- 压缩包中选取的七个源文件已原样解压到 `.work/ascendc_audit/proto/ascend_c/`，便于复核；没有修改这些源码。
- 以下 `tools/`、`kernels/` 路径均相对于该原型目录。
- SSH BatchMode 检查未通过认证；未使用密码登录，未检查服务器修复版，未执行 NPU 编译、精度或性能测试。
- 压缩包中的旧 GM 访问、L0 缓冲复用仍存在，且 CMake 指向的部分文件实际在 archive 下。不能将该快照当作服务器已修复版本。
- 用户提供的 108.28/851.49/3168.31 ms 与 Triton 对照属于历史测量，本轮没有新增性能数字。

## 已确认的代码问题

### 1. 实际 forward 仅使用两个 AscendC 计算阶段

`tools/kda_ascendc.py:43,63` 调用 gate/operand vector kernel 和 Aqk/Akk Cube kernel；K1 的求解、w/u，以及全部 K2 递推都在 NumPy 中执行。该入口没有调用 `k2_loop_chunk` 或其他 K2 AscendC 实验 kernel。

因此，第一优先级是迁移主机计算，而不是根据总体耗时调整 Cube tile。

### 2. 大量 D2H，以及三个完全未消费的拷贝

`tools/kda_ascendc.py:50-57,67-68` 共出现十处 `.cpu().numpy()`；其中 `qqg_n`、`kgk_n`、`kgq_n` 赋值后未使用。

每个无用结果为 `[B*H*NT,16,128]` FP32（MHA）。B=1、T=8192、H=32 时，三个结果合计 384 MiB D2H，不含 NPU 侧 BF16→FP32 的额外临时张量。这是当前 host 路径可直接清理的开销，但清理后仍保留 CPU K1/K2。

### 3. 声称 FP32 的 host 路径实际上引入 FP64

`tools/kda_ascendc.py:69-73,81,89-90` 的 `np.zeros`、`np.ones`、`np.eye` 没有指定 dtype。它们默认使用 FP64，并使求解、w/u、输出及默认初始 state 路径发生双精度计算。

最后 `torch.tensor(h, device=DEV)` 未明确指定 FP32；在默认 FP64 state 路径上，它不符合 docstring 的 FP32 返回约定。h0 是否给定及其 dtype 还会改变 host state 的舍入行为。

应先定义统一的数值契约，再改变 dtype：FP32 state 累加、BF16 Cube 输入，以及 w/u、v_new、qg/kg、Aqk 的舍入位置均需要与基准对齐。不能把切换 FP64→FP32 当作无精度影响的改动。

### 4. K1 每个 chunk 调用通用矩阵求逆

`tools/kda_ascendc.py:81` 对 `I+L` 调用 `np.linalg.inv`，而 L 是 16×16 严格下三角矩阵。

设备端候选方案：

- FP32 前代直接求解 `(I+L)W=Rk`、`(I+L)U=Rv`，避免显式保存逆矩阵；其舍入路径需独立验证。
- FP32 前代构造 16×16 的 A，再接现有 Cube w/u 原型，更接近当前阶段分解。

先实现并验证其中一种，再测量是否值得改为另一种；不能仅因 Cube 理论吞吐高就采用额外矩阵乘法。

### 5. `k2_loop_chunk` 不是完整 persistent kernel

`kernels/k2_loop_chunk.cpp` 没有 NT 参数、跨 chunk 时间循环，也没有 batch/head 的 block 偏移。文件头提到 graph-captured，并不代表实际入口已完成图捕获或设备端时间循环。

实际做的是一组 d1/d2 → v_new → d3/d4 → output/state。若外部每 chunk 调用一次，启动次数随 T/16 增长；不能把注释当作常数次启动的证据。MIX 1:2 代码也未见 AIV subblock 分工，需要核实重复写入和 flag 消费者数量。

### 6. K2 实验代码有确定的局部缓冲容量错误

- `kernels/k2_loop_chunk.cpp:172`：`ubVb` 分配 `16*128*2 = 4096` 字节。
- 第 278 行：`Cast(tVb, tH, ..., 128*128)` 要求 BF16 输出容量 32768 字节，是分配容量的 8 倍。
- 第 282 行：BF16 state 的 `DataCopyParams(16,128,0,0)` 按普通 DataCopy 的 32 字节块单位表示 65536 字节，完整 BF16 state 实际只有 32768 字节。

该 AIV 分支所有显式 InitBuffer 合计 194048 字节，尚未计额外编译资源；是否超出目标 UB 要依据实际 SoC/CANN 判断，不能仅凭此总数断言已经超出。

建议使用专用 state BF16 tile，并按 16 或 32 行分段转换/搬运；或让 K2 采用 BV=64 的 state 分片。BV=64 时单份 FP32 state 为 32 KiB，但仍须列出所有同时存活的缓冲、队列深度和对齐开销。

### 7. 接口与构建存在语义漂移

- `kda_forward` 接受 `lower_bound`，但没有传给 kernel；archive vecop 固定 LB=-5。
- 入口没有内部 Q/K normalization；若外部归一化，必须作为明确输入契约，并纳入公平的端到端计时。
- `_flat` 要求 T 为 16 的倍数，入口缺少清晰校验。
- Cube 源码注释称 Aqk 已乘 scale，但实际 `scale` 读入后未用于运算；Python 在 K2 再乘 scale。迁移时应明确 Aqk 是缩放前还是缩放后，避免重复缩放。
- `kernels/CMakeLists.txt` 引用根目录中的旧 kernel 文件，压缩包中相应文件在 archive；需先与服务器修复版对齐再建立可复现构建。
- 当前工作树 Triton `_run` 已针对 NPU D=128 选择 BV=64；这不能证明上述 AscendC 实验 kernel 也已分片。

## 三种路线

| 路线 | 内容 | 用途与取舍 |
|---|---|---|
| Host 清理 | 删除无用拷贝、清理无用数组、明确 dtype、优化小矩阵求解 | 小改动，能改善原型耗时；仍不能代表全 AscendC 性能 |
| 分阶段完成全设备 AscendC（推荐） | 设备 K1 solve/w/u，设备 K2，随后 K2 persistent 与流水优化 | 可逐阶段对照数值和性能，最终满足全 AscendC 目标 |
| Triton K1 + AscendC K2 | 复用稳定 K1，独立验证 AscendC K2 能力 | 最快隔离 K2 收益；应明确标为混合后端，可作为第二路线的临时验证桥梁 |

## 推荐实施顺序与验收

### 第一步：建立可复现基线

以服务器隔离目录中实际通过 smoke 的源码为基准，核对源码哈希、构建输入、SoC/CANN、launcher 和加载路径。统一输入布局、normalization、scale、lower_bound、h0 与 final_state dtype。

修复 K2 容量问题必须单独做小规模验证；不要将有越界的实验 kernel 接到长序列上。先确认单 chunk 非零 h0，再检查两 chunk 递推及多 batch/head。

### 第二步：设备端闭环

K1 保留 `(batch,head,chunk)` 并行，完成 normalization/gate/beta、Aqk/Akk、三角求解和 w/u。最初允许分成多个 kernel，通过同一 stream 与设备 workspace 传递数据。

K2 先完成正确的设备链路，消除 NumPy state/output 计算，明确每个任务的输入/输出偏移和 scratch 所有权。只在确有数据依赖的地方同步。设备闭环验收：输入和结果都在 NPU，forward 内无 `.cpu()`、`.numpy()`、`.item()`，无按 head/chunk 进行的主机数据处理。

可用已有 Triton K1 提供中间量独立验收 AscendC K2，避免把 K1 和 K2 的误差混在一起。

### 第三步：K2 常驻状态

在 K2 内部遍历 NT，每个逻辑任务拥有一个 `(batch,head,value_tile)` 的 state，推荐先以 BV=64 做容量安全的起点，再与 BV=128 实测比较。

最终 state 仅在序列结束按需写出；FP32 state 跨 chunk 保留在片上，Cube 所需 BF16 副本按生命周期转换。完成正确性后再减少 Cube/Vector 之间的 GM scratch。

目标是启动次数不随 NT 增长；总计算时长仍受递推依赖影响，不能据此承诺固定延迟或特定倍数。

### 第四步：缓冲复用和流水

- d1=w@h^T 与 d2=qg@h^T 共享相同旧 state，候选优化为复用 L1/L0B，或合并为 M=32 的一次乘法；必须等待前序读操作完成后才能覆盖缓冲。
- d4 的八个 M tile 使用相同 kgT，可尝试只加载一次 B；tile 数随 BV 调整。
- 为 Cube/Vector 明确生产者、消费者和搬运完成事件，再替换不必要的 PIPE_ALL。不能直接全局删除 barrier。
- 保留 K1 的 chunk 并行；K1/K2 全融合会改变并行度，不作为第一方案。

华为 MatmulPolicy 文档提供全载和缓冲策略的接口说明及约束，但对本项目是否有效仍需目标环境实测：
https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0619.html

### 验证口径

- 精度：单 chunk、两 chunk、B>1/H>1、非零 h0、默认及非默认 scale/lower_bound；明确尾块支持或拒绝。检查 finite、output/state dtype、最大绝对误差与相对/RMS 指标；阈值以统一参考实现和现有通过测试为依据。
- 逐阶段比较 Aqk、w/u、v_new、state，再扩展到三组历史长序列。仅 finite 不是完整正确性证明。
- 性能：预热后分开记录设备事件时间与包含布局转换、分配、normalization、state 输出的端到端墙钟；编译和首次加载单列。重复测量并报告中位数与波动。
- 原有 `test_ascendc_profile_fast.py` 只抽取少量 chunk 并线性外推 CPU 耗时，还漏计部分完整路径开销；可用于诊断，不能作为完整端到端 benchmark。

本轮交付是代码审查与候选设计，未更改运行代码，未声称获得加速。下一次实施应从版本对齐与设备端闭环开始。

## 服务器复核与实测（同日追加）

### 环境与版本

已通过交互式 SSH 登录用户指定服务器并进入 `chj_moonep_a3`，密码未写入脚本、命令历史或文档。检查和测试使用 `/workspace/kda_archive_20260905`。

- Torch：2.12.0+cu130；`torch.npu.is_available()` 为 True。
- triton-ascend：3.2.2。
- `torch.npu.get_device_name(0)`：Ascend910_9382。
- `ASCEND_HOME_PATH`：`/usr/local/Ascend/cann-9.1.0`。
- `npu-smi info` 所列设备健康状态均为 OK，查询时无运行中的 NPU 进程。
- 隔离源码树内未找到 AGENTS.md。

服务器 `tools/kda_ascendc.py` 与 `kernels/k2_loop_chunk.cpp` 和本地压缩包快照 SHA256 完全一致。这确认了 host K1/K2、无用 D2H、FP64 和实验 K2 容量问题仍存在。未运行有容量缺陷的 `k2_loop_chunk`。

实际构建的 vecop 已采用 `gExpA.GetValue(hv)`；Cube 第二次乘法已使用独立 L0A/L0B。服务器根目录 kernel 路径通过软链接指向 archive，故先前压缩包缺失的路径在服务器上已补齐。

| 文件（相对 proto/ascend_c） | SHA256 |
|---|---|
| tools/kda_ascendc.py | 33c8908cfb3c2e29b253b8da2cecdffbe06fbebe147e4fbae3bbdf764f17683c |
| kernels/k2_loop_chunk.cpp | be3c417e796dabc1ec84e9a466c1672ae9d91562a41e1fbec796aaa385a0a496 |
| kernels/archive/kda_k1_vecop.cpp | 4f7220e664adbfbb01d9b546f619b348b274f3d1b070dd5c42bab177a30be502 |
| kernels/archive/kda_k1_cube.cpp | b3e3d269837dc7f3c4b275f680768b23bbfcf7a328a3ecb441cf3e2640511c84 |
| kernels/build/lib/libkda_bt16_kernels.so | 2db950686f60ef210549865a3bd7b4595086df35de479cb2bf2444225c1b3e39 |

特别注意：旧 `/workspace/kda/proto/ascend_c` 实际指向生产目录 `/workspace/kda_final_20260905/aclab`。本次 smoke 在内存中将测试脚本 ROOT 指向隔离目录；forward 在加载后覆盖模块 LIB 为隔离目录下的动态库绝对路径，没有改动磁盘源码、软链接或生产目录。

### Smoke 复测

运行隔离目录 `tools/test_smoke.py` 的既有 P1→P3→P5→P6 顺序，保持原预热及参考计算。结果打印精度为小数点后六位：

| Probe | 最大绝对误差 |
|---|---:|
| P1 | 0.000004 |
| P3 | 0.000001 |
| P5 | 0.000004 |
| P6 | 0.000008 |

### Forward 墙钟基线

每个形状固定 seed=42，Q/K 为 NPU BF16 随机输入，先以 FP32 计算 `x*rsqrt(sum(x*x)+1e-6)` 再转回 BF16。V 为 BF16 随机值×0.3，g/beta 为 FP32 随机值×0.5，A_log/bias 为 FP32 随机值×0.1；scale=128**-0.5、默认 lower_bound=-5、h0=None。使用 npu:0，每形状预热一次，计时三次并在每次 forward 后同步。

计时包含原始 forward 中的布局转换、分配、CPU K1/K2、D2H 与输出/state H2D；输入生成和外部 Q/K normalization 不计入。它是此输入契约下的 host 原型基线，不能与 normalization 计入方式不同的历史数字直接作加速比。

| B,T,H,D | 三次耗时（ms） | 中位数（ms） | finite | final state dtype |
|---|---|---:|---|---|
| 2,1024,4,128 | 104.783 / 99.542 / 98.976 | 99.542 | True | torch.float64 |
| 2,4096,8,128 | 782.624 / 786.883 / 776.812 | 782.624 | True | torch.float64 |
| 1,8192,32,128 | 3084.857 / 3084.334 / 3078.539 | 3084.334 | True | torch.float64 |

未重新运行这些长序列的完整数值参考比较；finite 只证明没有 NaN/Inf。此次也没有同输入 Triton 性能对照，所以不报告新的后端加速比。

### 最大形状的阶段耗时

在独立进程中使用同样输入生成方式，对 forward 源码的第 25、49、59、69、89、112 行前仅插入墙钟时间标记，代码只在内存中执行，未改变算法。预热一次后测两次；下表为两次均值。各阶段使用入口已有同步边界，返回后额外同步。它们是墙钟阶段时间，不能称为纯 kernel 设备时间。

| 阶段 | 均值（ms） |
|---|---:|
| 布局转换、分配、vecop 与同步 | 2.643 |
| 输入转 FP32、beta sigmoid 与 D2H | 95.337 |
| Cube1、分配、同步与 Aqk/Akk D2H | 3.571 |
| Host K1：分配、求逆、w/u | 1345.769 |
| Host K2：分配、递推、输出 | 1768.754 |
| 结果构造/H2D、返回清理及最终同步 | 15.626 |
| 总计 | 3231.701 |

阶段测量的两次总时长为 3234.374、3229.028 ms，属于另一独立运行，不能强行与前表 3084.334 ms 的中位数拼接。CPU 负载、分配及其他运行差异均可能影响墙钟。

K1+K2 主机阶段合计约占 96.4%；全部输入 D2H 所属阶段约占 3.0%。三份无用拷贝只是该 3.0% 中的一部分，因此仅删除它们不可能消除数量级差距。

后续优先完成设备端 K1 solve/w/u 与设备端 K2，再做 persistent state；K2 可暂用 Triton K1 中间量独立验收。此次没有实施运行代码优化，所有实测数字均为现有原型基线。
