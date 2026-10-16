# 有界prefill V2控制greedy全文门槛失败：未做候选和吞吐

真实control owner记录configured_use_v2=True、WorkerFL、page16、MiniCPM 2048/42/16/2、asyncTrue、131072上限和2048预算。守卫已匹配，但control未启用任何新调度，deferred=0。

C1固定seed探针两次相同。两组C8并发4096输入/16输出、greedy、同8个输入seed和api seed777，16响应全部成功；索引0/2/4/5文本不同，其余4条相同。原脚本要求所有greedy全文重复相同，在control便失败，因此吞吐0轮、candidate未运行，无比较结果。这不是候选性能负例，也不能由这组数据定位差异成因，或否认已有独立Math102/105。

V3保留全部greedy比较作为诊断，改以实际恢复的GPU token/positions与canonical history精确相等检查调度恢复正确性，质量仍需新Math。V2原源码和failed状态未改为通过。

本机独立核验69源码快照/当前仓库源、16并发greedy响应的seed/temperature/usage/正文及差异索引、实际guard配置和空闲GPU记录。完整原包4414497 bytes、17分片SHA256 d9ce2027291f0e5facc55fda3aa4f31ad9fa06f36aa89fba64c2658eae70a87a 全部verified，见service-results/20261016_tianshu/prefill_cadence_failed_v2。无新Math或正式成绩，最好4093.81/Math102不变。
