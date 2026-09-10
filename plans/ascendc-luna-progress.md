# AscendC Luna 执行进度

主方案：[ascendc-luna-execution-plan.md](ascendc-luna-execution-plan.md)

更新时间：2026-09-08。服务器隔离执行根已完成 S00–S04、S12–S15；本地记录跟随服务器证据更新。

| 步骤 | 依赖 | 状态 | 验收证据 |
|---|---|---|---|
| S00 独立快照与环境 | 无 | passed | 服务器results/S00已有manifest、smoke与快照；缺失的文档附件仍需补齐 |
| S01 合同、参考、测试工具 | S00 | passed | 服务器results/S01/contract.json、triton-crosscheck.json |
| S02 launcher / 转置 / Cube微测 | S01 | passed | 服务器results/S02/cube.json、build-manifest.json |
| S03 设备预处理 | S02 | passed | 服务器results/S03/preprocess_check.json、preprocess.json、build-manifest.json |
| S04 FP32 Gram | S03 | passed | 服务器results/S04/gram_check.json、gram.json、build-manifest.json |
| S05 solve / w,u | S04 | not_started | — |
| S06 分离K2链路 | S01,S02 | not_started | — |
| S07 全设备闭环 | S05,S06 | not_started | — |
| S08 MIX单步 | S07 | not_started | — |
| S09 persistent GM-state | S08 | not_started | — |
| S10 FP32 state驻留 | S09 | not_started | — |
| S11 性能优化与公平对照 | S10 | not_started | — |
| S12 打包与干净重建 | S11 | not_started | — |
| S14 MIX d3/d4/output/state | S12 | passed | docs/ASCENDC_S14_MIX_AIC_RESULT_20260908.md |
| S15 MIX d12/vnew | S14 | passed | docs/ASCENDC_S15_D12_VNEW_RESULT_20260908.md |

## 当前交接

- 下一步：优先实现跨 chunk 的 persistent K2 pipeline，再推进设备端 K1 solve/w/u；S15 d12/vnew 仅保留为 opt-in 实验路径。
- 本地工作区：`D:\project\kda_final_20260905`。
- 服务器修复版：`/workspace/kda_archive_20260905/proto/ascend_c`。
- 容器：`chj_moonep_a3`。
- 新实现目录：`/workspace/kda_ascendc_luna_20260905`，已存在，包含baseline、S01工具和PERF旁支。
- SSH前一轮已正常退出；重新连接时通过交互式密码提示认证，不在文档保存密码。
- 本地无git remote且有用户未提交修改；独立目录模式。
- 既有证据：`docs/ASCENDC_OPTIMIZATION_REVIEW_20260905.md`，其中服务器追加节记录smoke、哈希、阶段时间。
- 本轮服务器已重新检查 NPU/CANN；当前执行根使用 caller NPU context 和 runtime aclrtc 编译。
- S03 设备预处理覆盖 B=2、H=3、NT=1/2/4、bias有无、LB=-1/-5、zero Q/K、输出canary；BF16/FP32中间量逐项通过。
- S04 使用 AIV FP32 pair Gram 建立正确性基线；Aqk32/L 均为显式下三角，Aqk16 单独 BF16 cast，canary通过。
- 容器 CANN 9.1.0 的离线 nested merge_obj_text.sh 对 AIV relocatable object 报 unknown file type；全局脚本已恢复，S03/S04以 aclrtcCompileProg 运行时编译验证，清单已记录该限制。
- 最新证据：`docs/ASCENDC_SERVER_REVIEW_20260907.md`、本地`.work/server_review_20260907/evidence.json`、服务器`results/REVIEW_20260907/`。
- PERF旁支：已有FP64 K1三角求解+FP32 NumPy K2候选；小形状约2.2–2.3×相对旧host加速，NPU重复性/独立精度未准入，不计入S02–S12完成。
- 生产目录 /workspace/kda_final_20260905 和旧软链接未修改；本轮仅清理了新执行根内由本轮产生的失败构建目录。

## 步骤日志模板

```text
日期 / step / substep：
状态：
源码目录和hash：
库路径和hash：
修改文件：
验收命令与退出码：
JSON / 日志：
通过 / 失败 / 跳过：
数值结果与dtype：
性能结果及口径：
问题 / 最小复现 / 已排除假设：
下一条命令：
checkpoint：
```

## 变更记录

- AMEND-002（2026-09-07）：根据服务器实际进度更新本地交接，钉住已修复BV=64参考hash；识别数学参考exp底数和g_local检查问题。详见主方案头部及服务器复核报告。当前计划变更不表示对应实现修复已完成。
- SERVER-S03/S04（2026-09-07）：服务器隔离根完成设备端 preprocess 与 FP32 Gram 基线；证据和限制见服务器 plans/ascendc-luna-progress.md 及 results/S03、results/S04。
- SERVER-S14（2026-09-08）：完成 `mix_aic_1_2`，AIC 融合 d3/d4，AIV 完成 output/state；H=32 使用局部 cross-core flag，性能与 `cube_full_d4` 基本持平。
- SERVER-S15（2026-09-08）：完成 `mix_d12_vnew`，并在 standalone 与 MIX d12 kernel 中加入 d1→d2 L0 复用 barrier。T=16/H=2、T=64/H=2、T=16/H=32 中间量、output、state 全部与 `cube_full_d4` 逐元素一致；四组长序列基准显示 S15 相对 S14 MIX 为 -1.7% 到 +0.9%，未形成实质收益。交付包 `kda_ascendc_S15_d12_vnew_20260908.tar.gz`，SHA256 `0ca35384707f11c88695dcc8ebb5776789670e812c791a9f93af8be2e27273f9`。
