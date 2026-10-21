# BM8候选实际C64 Math完成，按用户要求停止

2026-10-21 22:56，105道MATH-500 Level3完整完成，103/105正确，准确率98.095238%，满足>95%的数值门槛。batch64、temperature1、top_p0.95、max_tokens32768，105个预测与评分逐条复核，0评测请求错误。没有重跑、重抽题或缩短生成上限。

用户要求“跑完math-500就停”。22:57已关闭服务并结束后续调度器；ixsmi显存68MiB、利用率0%、无运行进程。未启动后续官方C8 Math、正式4K N256或16K，也不继续其他优化。

原控制器保留failed/AssertionError。原因是我在运行中归档短测CSV时，将非数值依赖KEY_ROUNDS_32g.csv从CRLF改为LF，导致末尾controller_sources校验失败。这个操作不应在依赖冻结后发生。500个模型数值源码和其余控制器依赖完全未变，evalscope退出0，105题结果完整。原CSV已从运行前快照恢复到原始字节和原哈希；失败时LF文件与原失败state/result均保留。结果目录新增.gitattributes固定该CSV字节，避免Git换行规范化再次破坏精确依赖绑定。

独立CPU审核在GPU停止后完成，禁止导入torch/triton/vllm/flag_gems：500个原数值源码、14个原控制器哈希、105条预测与评分、实际C64请求的42层新BM8捕获与Graph replay、M64 GateUp/LM整数路径、M4096全42层融合均通过原冻结审核函数。仅重新核对原结果，没有新GPU请求。C64_CPU_RECOVERY_GATE_32g.json记录原失败状态未改写及所有证据；这不能宣称原控制器成功，也不能替代官方C8质量门禁。

短测4K/1K、C64/N64后两轮均值5753.04 total、1150.61 output tok/s，相对同批原版5616.42提升2.4325%，后两轮差幅0.1955%。这是N64短测，不能称为新的正式N256成绩。候选仍未采用。

正式推荐版保持5695.95 total、1139.19 output tok/s，TTFT3663.77ms，tag为tianshu-original-command-4k-m-range-fused-20261021。后续只有用户明确恢复后才继续；若推进BM8候选，需在相同500数值源码/配置上绑定此次独立C64审核的新控制器版本，完成官方C8 Math，再测正式4K。

原始完整响应、评分、日志和冻结快照仅留在容器/workspace/benchmark-results/20261021_BM8_v3_c64_quality_tianshu；原失败state/result原封不动。独立CPU完整审核在/workspace/benchmark-results/20261021_BM8_v3_C64_CPU_recovery_tianshu。GitHub仅保留代码、报告和关键门禁证据。
