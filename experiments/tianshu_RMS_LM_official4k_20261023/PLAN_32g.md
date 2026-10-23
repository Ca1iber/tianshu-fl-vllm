# RMS与LM组合：仅在新鲜双质量及同期短测通过后运行完整4K

本目录准备完整官方4K入口，当前尚未通过短测、没有正式GPU成绩。原13serve参数与63环境，944模型输入，原r3的903项精确子集。C8/C64均是此组合首次新鲜完整105题，必须严格>95%并经独立CPU逐题及全量旧/新审计核验；随后同请求同期r3对照N64×3排首轮增益>0.5%、候选热轮spread<1%、平均TTFT<=control1.01，才允许本入口启动。

HOLDOUT及--screen-root使用本组合质量实际的tianshu_LM_pad64_20261022，明确与旧RMS单独质量的参考根区分；不伪造matched_screen_root。运行原benchmark_throughput_serve.py完整原脚本，仅[[4096,1024,64,256]]，4轮排第1轮，1024请求全部成功；不修改请求/词表/输出上限、不跑16K。GPU互斥，最低剩余2400秒，2026-10-23T15:58Z硬停。

静态M64 RMS+A8与LM pad32..64实际图捕获/二进制/pack/profiler审计，全部旧attention/KV/scheduler/sampler/GU/Down/QKV/O审计保留；旧独立A8和旧LM64执行不冒称，完整比较独立CPU重算须逐字段相同。正式提升门槛为>5950.75×1.005 tok/s、后3轮spread<1%、TTFT<=3174.58×1.01，零preemption/服务ready后现场调优。未过则保留未采用版本。原模型checkpoint及结构不变，无自动发布。

原始请求/响应/日志/快照/数组/缓存仅天数容器，代码/报告/关键结果GitHub。
