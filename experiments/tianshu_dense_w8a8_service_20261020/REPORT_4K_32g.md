# 选择性 W8A8：小规模4K完整对照

2026-10-20，v4恢复GateUp非INT8批次的已测BF16 NT mm回退后，两组均通过独立审计。4096输入/1024输出、C64、N64，各3轮排首轮：control5175.23→candidate5308.68 total tok/s，提升2.5786%。这不是N256正式成绩。

|模式|轮次|total tok/s|output tok/s|mean TTFT ms|现场调优|计分|
|---|---:|---:|---:|---:|---:|---|
|control|1|4978.59|995.72|13670.77|0|排除|
|control|2|5181.52|1036.3|11499.15|0|是|
|control|3|5168.95|1033.79|11519.77|0|是|
|candidate|1|5129.48|1025.9|13533|0|排除|
|candidate|2|5300.44|1060.09|11542.95|0|是|
|candidate|3|5316.92|1063.38|11521.05|0|是|

吞吐请求384/384成功，另有C8短诊断16个、C1确定性诊断4个，均无请求错误。计分两轮极差/均值control0.2428%、candidate0.3104%，六轮均无新增现场autotune。每轮输入前缀按相同seed配对，两组采样参数/原13项argv一致，只改变TIANSHU_DENSE_INT8_MODE；311份来源和57项实际进程环境均核验。

实际模型只打包42个GateUp与1个ParallelLMHead。新增权重/scale1326938112bytes，保留原BF16权重供其他批次使用；打包在实际活跃的FL模型DeviceMemoryProfiler内完成。控制组KV blocks44129，候选40731；没有隐藏权重占用，完整16K影响尚未测量。

CUDA Graph中C8/C64各42个GateUp整数分支真实捕获并关联实际模型回放；输出头在model Graph外独立执行整数GEMM并验证调用。layer0 GateUp与LM头的原始BF16、INT8、scale哈希与已通过组件测试的权重一致，输入embedding未打包。C1确定性文本在组内和组间一致仅作诊断，不代替新质量。

两组统一VLLM_DISABLE_COMPILE_CACHE=1以避免旧AOT图绕过新GateUp；未清理全局编译缓存。前三次接入问题见STARTUP_AND_BINDING_FAILURES_32g.md，快照与日志保留。本次没有放宽基线GEMM、KV、采样或实际分支审计要求。

同311份来源/57项环境的新105题MATH-500已排入独立流程，必须严格大于95%后才继续原始完整4K四轮N256；没有复用先前103/105。本报告未确认新质量或正式吞吐，adopted=false。当前正式5235.52版本继续保留。

原始目录：/workspace/benchmark-results/20261020_dense_w8a8_service4k_v4_tianshu。完整回答留容器，Git保存关键统计、权重哈希与源码证据。
