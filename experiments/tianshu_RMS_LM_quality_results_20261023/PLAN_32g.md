# RMS与LM新组合完整双质量独立CPU核验

模型944、质量控制器91，原RMS916与LM925完整源均为精确子集；重新核验全部源码/前快照、原r3与checkpoint SHA。只用本组合新C64/C8各完整105条原始review/prediction/report计分，cap/errors全计入、不借父分数、不原样重跑。

在首次组合GPU前固定此核验。重新执行所有原框架/attention/KV/scheduler/sampling/GU/LM/Down/QKV/O及新RMS融合与LM范围/权重/Profiler/二进制检查，完整profile和accuracy_audit与本组合原控制器结果相等。核对实际42层RMS/GU捕获及兼容索引未伪造旧A8、LM64旧计数不再声称、全部深层LM守卫已明确替换为新真实计数、动态范围与其余路径保留；C64需实际新LM64调用及各42层M4096覆盖。原13serve参数/63环境、105题分母、GPU结束空闲均要求。

单批次partial Gate不能授权benchmark；两组全新正确率严格>95%且全部独立审计通过才model_quality_qualified。代码只CPU不导入Torch，不改原始数据或model。随后独立短4K与当时最佳合格版同期对照，过筛才正式，无16K或自动发布。原始数据/数组/日志仅容器，GitHub只关键分数门槛和报告。当前未运行组合GPU，故无新组合质量结论。
