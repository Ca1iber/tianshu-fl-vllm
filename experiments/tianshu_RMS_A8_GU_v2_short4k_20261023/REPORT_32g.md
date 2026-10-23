# RMS+A8 GateUp v2：质量通过，短4K提升0.1344%，不进入正式测试

| 版本 | 第1轮（排除） | 第2轮 | 第3轮 | 后两轮平均 total tok/s | 后两轮平均 TTFT ms |
|---|---:|---:|---:|---:|---:|
| 原r3同期对照 | 5510.33 | 5848.81 | 5902.61 | 5875.71 | 8569.58 |
| 静态M64 RMS+A8融合 | 5501.51 | 5859.74 | 5907.47 | 5883.61 | 8600.30 |

六轮各64请求全部成功，每组运行三轮，事先约定排除首轮。候选均值增幅0.1344013%，热轮极差/均值0.8111870%，平均TTFT增加0.3585%；通过波动及TTFT条件，未达到预先设定的>0.5%吞吐条件。SCREEN_GATE.complete=true、candidate_passes_perf_screen=false，不运行该版本完整N256 benchmark，不采用或发布。

本版本916项模型输入、原13serve参数及63环境在首次GPU前冻结；原r3的903项是字节精确子集。只融合静态M64 GateUp前42层IR-native RMSNorm与A8量化，后续原GU GEMM与SiLU保持；QKV64/O64/Down64和LM等保留r3路径。全部既有与新增运行审计及真实图捕获、norm/GEMM CUBIN、原checkpoint/source snapshot检查通过。旧独立A8没有执行，兼容统计只记录实际相同GU GEMM，未伪造量化器调用。

新鲜独立Math-500 Level3为C8 103/105、C64 101/105，两批均严格>95%，包括达到输出上限的请求，全部105题均计入。完整独立质量证据在tianshu_RMS_A8_GU_quality_results_v2_20261023，不能以组件25.69%延迟降幅作为模型吞吐收益。

正式最佳仍为r3的5950.75 total tok/s、1190.15 output tok/s、TTFT3174.58ms；本表N64结果不与N256正式成绩直接比较。接下来单独验证RMS与LM pad32..64的组合，重新做完整质量及同期性能对照，不把两项小收益直接相加。

代码、报告、关键分数和门槛放天数GitHub，完整请求、日志、数组、快照及缓存仅留天数容器/workspace/benchmark-results/20261023_RMS_A8_GU_v2_short4k_{control,candidate}_tianshu。本地没有下载副本。
