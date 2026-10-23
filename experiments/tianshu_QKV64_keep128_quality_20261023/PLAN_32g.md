# 新QKV64 keep128双质量控制

严格继承上一uniform64质量控制器的全部原审计，唯一数值路径差异为恢复原r3的QKV128 native；新独立service/policy/audit和所有质量控制输入在首GPU前冻结。C64先跑105题，严格>95%才新跑C8完整105题，不复用903/915/928/925分数，不删生成上限或错误题。原13serve argv/63环境保持；C64仅按原质量对照声明将KV_MODE设control，不人为改KV字节预算。真实QKV64上下文、QKV128 native路由、84pack实际内存、六个warmups/CUBIN、Down与原全部审计须通过。不做其他GPU任务，今日23:58硬停止并核验空闲。资源预检至少2GiB，无质量/数值门槛放宽。
