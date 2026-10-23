# RMS与LM独立组合：全新双批次质量，准备阶段

模型944项：原RMS916与LM925完整源集合的并集及新胶水/门槛；不改父数值kernel。当前RMS v2完整新质量C8 103/C64 101均>95%只作为进入组合的前提，不复用为组合分数。组合必须新鲜完整C64/C8各105题严格>95%，所有cap/errors计入分母；无同版原样重跑取幸运值。原13serve参数/63环境，C64保留原FutureKV control政策。

GPU入口拒绝当前RMS quality18102或short序列19611仍活动、任何服务/其他试验或GPU非空闲，避免并行干扰；本目录不自动排队，当前RMS短4K先完成并审阅。新组合先同次服务startup audit再Math，不另起无意义重启。LM核心及math/official深层导入副本的旧LM64计数守卫在首次GPU前全部明确替换为真实新LM记录、所有其余检查保留，旧LM64执行不再声称；RMS兼容索引只说明真实相同GU GEMM，旧独立A8未执行。RMS记录里LM未改字段仅指RMS自身钩子范围，整体组合LM明确已替换且独立核验。

实际norm/GEMM CUBIN/42层图捕获重放、LM全部命中M/实际原pack/Profiler峰值、输入/权重/RNG、全源/控制器快照、所有原attention/KV/scheduler/sampling/Down/QKV/O/GU/M8与LM<=8 BF16检查保留。质量独立CPU逐题/全量运行审计通过后才另建短4K，N64×3排首轮、收益>0.5%、波动<1%、TTFT<=同期对照1.01才原N256×4正式。无16K、无自动发布；15:58Z硬停。模型/依赖/原始响应/缓存/数组仅容器，代码报告关键结果GitHub。
