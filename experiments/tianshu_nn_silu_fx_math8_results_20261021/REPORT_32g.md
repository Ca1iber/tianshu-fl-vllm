# 静态M64与NN两次dot融合候选：官方Math8结果

新鲜官方Math500 Level3 105题，batch8，temperature1/top_p0.95/max_tokens32768，原始13项serve命令、同屏测450数值源与62环境。101/105=96.190476%，全部105请求成功、0错误，严格大于95%，通过质量阈值。没有复用旧版102/105，也没有重跑挑分。

完成后CPU执行冻结framework_profile的全runtime与Math审核，检查真实M8 NN整数路径、BF16 LM、42层真实prefill INT8路径、图/权重/KV与启动预热来源；MATH_GATE保留关键审核结论与450源摘要，完整重复的逐层记录仅留容器并记录SHA256。原始题目回答和张量留在容器，GitHub仅关键审核/报告/摘要。

batch8不能证明M64融合在真实请求中重放后的质量。串行runner已开始同105题/max_tokens32768的C64补充，只把FutureKV准入candidate改为control以允许真实64活跃请求，保留原生allocator及450数值源。必须实际覆盖42层M64融合重放与LM64 INT8，并且再次严格>95%，才能跑正式N256 4K。

性能资格仍只是此前4K/1K C64 N64三臂屏测：原始407源5508.9690，静态64对照5529.4693，融合5554.9744 total tok/s；整体+0.83510%。本次Math不是吞吐新纪录，正式最佳仍5586.28 total /1117.25 output tok/s、TTFT3780.63ms。

下一独立组件40efaa4已冻结：K8V16 Q1/Q2直接3D，894组CPU映射、128token缓冲、安全串行启动门控通过；尚未跑GPU，不计入任何服务收益。
