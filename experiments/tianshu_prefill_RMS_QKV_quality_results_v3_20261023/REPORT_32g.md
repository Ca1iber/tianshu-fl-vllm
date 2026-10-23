# v3 M4096 QKV norm/A8融合：新鲜双组质量全部通过

冻结模型06bca86，925项模型来源/82项控制器；正式r3原903项来源逐字保留，原13项serve argv及63项环境契约核对。新增静态M4096 QKV前norm+A8融合；该范围重新接回相同原native GateUp，原通用/M64 pass实际记录仍86份，新范围43份另审计。未合入RMS64/LMpad组合，不修改checkpoint/模型结构或新增weight pack。

| 本次新鲜Math Level3 | 正确/总数 | 正确率 | 错误index | 截断/执行错误 |
| --- | --- | --- | --- | --- |
| C8 | 101/105 | 96.190476% | 15、18、26、87 | 0/0 |
| C64 | 101/105 | 96.190476% | 18、35、83、87 | 0/0 |

两组各105题全部stop，错误全部计入，均严格>95%。C64沿用既有FutureKV control质量覆盖政策，C8保持candidate发布策略。C64真实运行42层新融合及原QKV/O M4096各42层、Down全42层；C8实际未命中这些native attention/prefill新范围，不把C8启动预热冒称真实请求覆盖。两次启动均核验129份pass、原GU恢复42、新QKV替换42、无残差/有残差真实CUBIN与原NN指针；新scratch在原worker内存profile窗口显式probe计入，manual KV bytes None，RNG/原weight版本保持。

独立CPU全量重读210条review/prediction及两份report，重算全部旧/新runtime profile与accuracy_audit逐字段完全相等；925项模型来源、82项控制器的当前文件和冻结快照SHA及原checkpoint SHA重新核验。QUALITY_GATE的model_quality_qualified=true，KEY_SCORES保留逐题分数。原始回答/日志只在容器`/workspace/benchmark-results/20261023_prefill_RMS_QKV_v3_math{8,64}_tianshu`。

当前只确认质量通过。新的短4K对照已按顺序启动：control为原r3 903来源，candidate为925来源，各N64三轮排首轮，全部审计独立重算；>.5%吞吐、候选热轮波动<1%、TTFT<=对照1.01才允许完整N256四轮4K。本版尚无正式吞吐结论或采用，正式最好仍r3 5950.75 total tok/s。

按历史真实43008次QKV调用和3层组件绝对延迟外推，单靠本项prefill融合整体可能仅+0.111%—+0.122%；该数字是带假设的估算，非严格上界或整机测量，详见`tianshu_prefill_RMS_Amdahl_20261023`。短测结束后另行排队原生INT4常量微测及完整INT64参考的8个非恒定格式假设；此时尚无INT4 GPU或性能结论，不并发GPU、不跑16K。
