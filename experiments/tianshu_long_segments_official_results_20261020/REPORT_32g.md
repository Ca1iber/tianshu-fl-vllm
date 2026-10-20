# 长上下文S16正式采用：4K 5099.65、16K 3723.12 total tok/s

2026-10-20，天数BI-V150 32G。原始双负载benchmark已经完整结束；每个负载四轮，按脚本预设排除首轮，后三轮取平均。八轮共1536/1536请求成功，0失败。新Math Level 3为102/105（97.142857%），同一225份冻结源码、46项归一化环境和原始serve argv核验通过，真实FL Graph捕获与重放绑定核验通过。

| 正式负载 | 总tok/s | 输出tok/s | 平均TTFT ms | 每轮耗时 s | 三轮吞吐极差/均值 |
|---|---:|---:|---:|---:|---:|
| 4K/1K C64 N256 | 5099.65 | 1019.93 | 4198.32 | 257.02 | 0.11373% |
| 16K/1K C64 N128 | 3723.12 | 219.01 | 109729.32 | 598.48 | 0.06097% |

16K四轮总吞吐依次为3719.74（预设排除）、3724.31、3722.04、3723.02；每轮128/128成功。4K依次4970.96（预设排除）、5099.92、5102.41、5096.61；每轮256/256成功。完整八轮与正式汇总见[RAW_RUNS_32g.csv](RAW_RUNS_32g.csv)和[SUMMARY_32g.csv](SUMMARY_32g.csv)。

与前一已验证版本比较，4K 5101.45→5099.65，变化-0.03528%，基本持平；16K 3213.11→3723.12，提高15.87278%。这是前后两次完整评测的历史比较。严格的同负载N64服务对照为3386.67→3751.55（+10.7740%），见[对照报告](../tianshu_long_segments_results_20261020/REPORT_16K_32g.md)。相对官方基线，4K较2028.01提高151.46079%，16K较915.15提高306.83167%。

采用的新增改变仅为长上下文attention分段：GPU实际长度≤8192继续4活跃段，较长上下文仅B33..56选择maxS16；其它批次维持已有分段策略。保留K软件8位/V BF16、12KiB物理页、原生默认4096预算、83预热点、严格TopP32K、M2048 GEMM策略。没有加入正在独立验证的KV watermark或packed-load候选，也未修改模型权重、原始benchmark和沐曦仓库。

本次完整服务累计24次KV抢占。4K结束的观测快照为0次，16K结束累计24次；仍有进一步优化空间。服务就绪后记录到一次`_index_linearized_jit_function`索引调优，耗时0.51秒，没有观察到本次就绪后GEMM调优日志；这不是对所有现场调优的排除证明。报告没有用无效硬件profile推断占用率。

[OFFICIAL_AUDIT_32g.json](OFFICIAL_AUDIT_32g.json)和[OFFICIAL_GATE_32g.json](OFFICIAL_GATE_32g.json)保留原始测试判定与CSV来源哈希；其中原始adopted=false快照保持原样。[ACCEPTANCE_GATE_32g.json](ACCEPTANCE_GATE_32g.json)单独记录正式采用，要求新Math严格>95%、双负载完整成功、4K较旧版回退不超过1%、16K提高至少2%。该采用条件用于本项目版本决策。

发布标签`tianshu-original-command-4k16k-k8v16-longseg-20261020`，同配置启动入口见[release](../tianshu_k8v16_longseg_release_20261020/README_32g.md)。源/环境核验、原始请求与日志均在容器；GitHub仅保存代码、报告、关键JSON/CSV。完整原件目录为`/workspace/benchmark-results/20261020_long_segments_official4k16k_tianshu`，质量原件为`/workspace/benchmark-results/20261020_long_segments_math_tianshu`。恢复历史生成源码入口已经验证当前容器所需文件的SHA；新克隆整套环境安装与模型启动尚未验证。
