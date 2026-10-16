# cadence2持续补请求验证：吞吐+0.647%，未采用

原服务参数/current best算子、C64/N128、4K/1K，固定candidate三轮再fresh control三轮、各排首轮。后两轮control均值4071.712523、candidate4098.069748 total tok/s，+0.647325%。比N64的+0.9507%小；这是小规模筛选，非N256正式成绩，无新Math，暂不采用。

|arm|轮|total tok/s|客户端TTFT ms|新增调优|计统计|
|---|---:|---:|---:|---:|---|
|candidate|1|4011.976|8569.537|0|否|
|candidate|2|4100.337|6849.017|0|是|
|candidate|3|4095.803|6763.734|0|是|
|control|1|3984.944|8764.467|0|否|
|control|2|4074.433|7072.213|0|是|
|control|3|4068.992|7065.483|0|是|

全部6轮第二批HTTP请求都在第一批完全结束前实际进入。ThreadPool64中尚未开始的任务排队时间不计单请求TTFT；组总时长包含全部128请求，与前置说明一致。两个arm同轮seed相同，轮间2026101700+1000*i+j无重叠。全部768性能请求和32个并发canary原始文本/usage/种子/流时序保存，本地重算通过；四个C1 seeded相同，32个greedy canary跨arm12/16相同仅诊断，不当质量门槛。

客户端稳态TTFT7068.848→6806.376ms（约-3.71%），平均单请求wall79240.490→79416.562ms（约+0.22%）。实际server metrics按每轮before/after差分，逐轮核验128请求、524288 prompt tokens、131072 generation tokens、128 TTFT样本及130944 inter-token样本，稳态server TTFT7008.738→6750.171ms，平均ITL70.568→70.989ms（约+0.60%）。这比非空文本SSE间隔更适合描述token间隔；客户端最长文本事件gap36.00→41.21秒仍不是最长逐token时延，不能据此声称GPU或scheduler暂停40秒。

候选仍是有界cadence2+WorkerFL原MRV1 all_token_ids恢复兼容路径组合，不能把整个差额单独归因cadence。实际canary有10个held步/5个history payload步、21项恢复GPU token和position逐项一致，吞吐前解除同步arm、journal SHA保持；没有保存所有GPU Tensor。70源快照及本地所属源、实际原serve/default2048/256seq/75warm/40环境/CV-KV16-S8/strictTopP32Kgeo/三M2048 launch与16份metrics零抢占全部独立核验。

归档21,513,067字节、83分片，SHA256 bb243d2015dcf9905b467eb38283e252e50456311905db0d1ab6b16933f39eb5。原件service-results/20261016_tianshu/prefill_cadence_holdout_service/20261016_prefill_cadence_holdout_service_tianshu。当前正式最佳4093.81 tok/s、Math102/105不变。
