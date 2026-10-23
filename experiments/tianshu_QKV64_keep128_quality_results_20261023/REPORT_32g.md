# QKV64均匀整批原生INT8、保留QKV128：新鲜双批次质量结果

2026-10-23新模型输入928项、质量控制器51项均于首次GPU前冻结。该候选保留r3原903项模型输入的精确字节，只新增FULL/64tokens/64requests/均匀解码/无LoRA的QKV64原生INT8，非均匀/非图M64及O64仍BF16，QKV128继续原生INT8；没有合入LM925 pad候选。原MiniCPM5-2B checkpoint及结构不变，原13项serve参数/63项环境保留。

| 完整Math-500 Level3 | 正确/全部题数 | 正确率 | stop/max_tokens |
|---|---:|---:|---:|
| C8 | 102/105 | 97.142857% | 105/0 |
| C64 | 102/105 | 97.142857% | 103/2 |

两组均严格>95%。C64索引26、83达到输出上限，照常计入分母；错误索引为18、26、87，C8错误索引为10、18、87。没有删除超长题，也没有同版本重跑挑成绩。原始GPU控制器均phase=complete、evalscope退出码0，未使用旧准确率或CPU审计恢复路径。

独立CPU验证器重新读取逐题原始评审，重算两组全部105条，检查928项模型输入、51项质量控制器及其源快照、原checkpoint SHA、全部继承检查、真实84份NN打包与DeviceMemoryProfiler、QKV128保留和非均匀M64 BF16策略。C64实际QKV和O各42层执行M4096；C8实际QKV各42层执行M128。两组均捕获42层QKV64原生路径，O64捕获BF16；Python路由次数不当成图重放次数。独立QUALITY_GATE.complete及model_quality_qualified均为true。

该结果只使同一模型版本具备进入短4K吞吐对照的条件，尚无本版本正式性能或合格发布。随后按GPU前冻结规则比较r3与候选N64×3、排除第一轮；仅稳定收益通过时继续N256×4正式4K。未运行16K。当前最佳合格发布仍为r3正式5950.75 total tok/s。

原始输入、输出和完整日志仅在天数容器的/workspace/benchmark-results/20261023_QKV64_keep128_math{8,64}_tianshu；GitHub保留代码、门槛、逐题关键分数与本报告。
