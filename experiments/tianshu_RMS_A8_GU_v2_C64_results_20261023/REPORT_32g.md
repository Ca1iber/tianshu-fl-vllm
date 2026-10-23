# RMS+A8 GateUp v2：完整C64为101/105，独立全量审计通过

新鲜完整Math-500 Level3 C64 **101/105（96.190476%）**，严格>95%。105条请求全部成功，104条stop、1条max_tokens；索引101达到输出上限，照常计入分母。错误索引18、87、99、101，没有删题或同版本重跑挑成绩。

独立CPU验证器为完整双批次verifier的原字节副本，固定--batches64。重新读取全部原始review/prediction/report并计分，核对916项模型输入、80项质量控制器及其GPU前源快照，原r3精确903项子集及checkpoint SHA不变。全部继承框架/量化审计和新RMS+A8专项审计重跑，完整profile与accuracy_audit重新计算后均与原GPU控制器快照完全一致。

实际42层新RMS+A8捕获与原GU GEMM二进制验证通过，旧兼容索引明确为真实同GEMM kernel调用，原独立A8未执行，不伪造旧量化调用；动态范围保持原图字节。C64运行实际QKV/O各42层执行M4096，QKV另击中M128；QKV64/O64/Down64保留BF16。原13serve参数及63环境保留，既有C64 FutureKV control质量政策未改。

QUALITY_GATE.complete=true，但model_quality_qualified=false：此目录只验证单批次，不能据此授权benchmark。C8首次新鲜质量仍待完成；完整双批次独立核验通过后才运行预先冻结的N64短4K。没有该模型正式吞吐或新合格版本，r3仍是最佳正式版。

原始数据/workspace/benchmark-results/20261023_RMS_A8_GU_v2_math64_tianshu仅在天数容器。GitHub仅逐题关键分数、门槛、源码和报告。
