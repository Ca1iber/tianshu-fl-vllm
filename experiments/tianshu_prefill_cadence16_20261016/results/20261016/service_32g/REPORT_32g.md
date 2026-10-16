# cadence16：吞吐仅+0.543%，TTFT代价更大，不采用

原参数、current best算子、C64/N128 4K/1K，预定control先三轮、fresh candidate后三轮，各排首轮。候选只把cadence2的每3步最多2步defer改为每17步最多16步，其他兼容恢复/guard/finally不变。后两轮control4075.118759→candidate4097.226899 tok/s，+0.542515%，不优于先前cadence2的+0.6473%；两次控制不同，不能将候选绝对值跨配对直接当精确差异。未采用，无新Math/正式。

|arm|轮|tok/s|客户端TTFT ms|新增调优|计统计|
|---|---:|---:|---:|---:|---|
|control|1|3990.333|8744.077|0|否|
|control|2|4072.460|7067.488|0|是|
|control|3|4077.778|7073.589|0|是|
|candidate|1|4010.612|11943.974|2|否|
|candidate|2|4091.312|8428.479|1|是|
|candidate|3|4103.142|11710.171|1|是|

稳态客户端TTFT7070.539→10069.325ms，约+42.4%；平均请求wall79171.525→79654.854ms，约+0.61%。真实server计数差分：TTFT7013.265→9892.190ms（约+41.1%），ITL70.497→68.030ms；第一token更晚且wall略增，因此不能把较低ITL直接当整体响应更快。六轮每轮128请求、524288输入、131072生成、128 TTFT样本和130944 ITL样本均逐份核验。无绝对时延上界承诺。

候选调优全为_index_linearized_jit_function，而非GEMM：首轮raw M290/30，后两轮M32/29，分别日志0.21/0.16/0.18/0.15秒，int32源/int64索引/int32输出。相同align32 cache key仍可有不同raw benchmark key；现用75点是GEMM预热，不意味着所有index场景预热。没有从分数扣除此成本，也不能把多秒TTFT变化全归因这几项日志。

实际canary15个held步/3个history payload步、14次恢复GPU token/position与调度历史一致；吞吐前移除同步arm、journal SHA保持。4个C1 canary相同，并发greedy跨arm一致仅诊断不做质量门槛。全部768性能响应完整文本/usage/seed/时序及真实补队列、相同70源/40环境、原serve/default2048/256seq/75warm/.cv/KV16/S8/三M2048/strictTopP32Kgeo与16指标零抢占下载独立核验。

后续完整评测准备器在观测本次稳态前设置N128至少+1%、两轮均超过控制且极差<0.5%的性能门槛；对本次实际结果执行check_screen，明确因+0.5425%<1%拒绝，不启动昂贵新Math/正式。该1%是本轮ROI门槛，与官方正确率严格>95%分开；没有修改旧GEO>=2%门槛或复用旧准确率。当前正式4093.81/Math102/105保持。

归档21,195,801字节、81分片，SHA256 1aa929af2066b65fe3350fca8dab30ad3e00699faabc766bb8bc3b5e1d50b6e9；原件service-results/20261016_tianshu/prefill_cadence16_service/20261016_prefill_cadence16_service_tianshu。
