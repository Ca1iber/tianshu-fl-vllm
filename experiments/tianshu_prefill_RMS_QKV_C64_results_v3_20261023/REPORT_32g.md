# v3 prefill QKV norm/A8融合：C64新鲜质量通过，C8待完成

冻结模型06bca86，925项模型来源/82项控制器；原r3的903项来源全部逐字保留，原13项serve argv和63项环境契约核对。只新增M4096的QKV norm+A8融合，独立恢复新静态范围中原native GateUp，不含RMS64或LMpad组合。

2026-10-23本次C64全部105题：101/105=96.190476%，严格>95%；4个错误index为18、35、83、87。105题全部stop，0截断、0执行错误、0显存抢占。42层新融合均在真实请求中命中；原QKV/O M4096各42层、Down及其余原有框架/算子检查全部保留。无残差/有残差scratch确实在原worker内存profile窗口计入，manual KV bytes为None，原权重版本/随机数状态保持。

独立CPU重新读取105条report/review/prediction并重算全部旧/新runtime profile和accuracy_audit，逐字段完全相等；925项模型、82项控制器的当前文件及各自快照SHA和原checkpoint SHA均复核。KEY_SCORES记录每题结果；原始日志/完整回答只在容器`/workspace/benchmark-results/20261023_prefill_RMS_QKV_v3_math64_tianshu`。

当前只完成C64，model_quality_qualified明确为false。C8已按顺序启动，双组全部独立通过后才进行原r3同期短4K。尚无本版模型吞吐结论，未采用或运行正式4K；正式最好仍r3 5950.75 total tok/s。原生INT4微型探测排在质量与短测序列之后，不并发GPU，尚无INT4硬件/性能结论。
