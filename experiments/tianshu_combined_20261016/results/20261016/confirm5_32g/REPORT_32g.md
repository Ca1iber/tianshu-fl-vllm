# 天数 BI-V150 32g：独立五轮组合确认，吞吐+2.06%

新鲜5+5服务复测，4K输入/1K输出、C64/N64，每组预排前两轮。与screen相同23源，控制budget4096/83warm/maxseq256/native TopP/原采样/stream1；候选budget16384/131warm/maxseq64/严格TopP16384/8192/融合采样/stream4。两边CV/KV16/w1/S8、async=True、BF16和native Prefill不变。

|组别|轮次|total tok/s|output tok/s|TTFT ms|
|---|---:|---:|---:|---:|
|control|1（排除）|3800.80|760.16|17487.67|
|control|2（排除）|4070.14|814.03|11736.77|
|control|3|4048.32|809.66|11775.03|
|control|4|4068.44|813.69|11768.73|
|control|5|4080.75|816.15|11762.14|
|candidate|1（排除）|3867.79|773.56|16662.40|
|candidate|2（排除）|4122.98|824.60|12383.96|
|candidate|3|4160.40|832.08|12200.43|
|candidate|4|4144.25|828.85|12607.01|
|candidate|5|4144.21|828.84|12656.76|

后三轮均值4065.8366→4149.6193 total tok/s，提升 **2.060652%**，各候选计分轮均高于控制计分轮。平均TTFT 11768.63→12488.07ms，增加 **6.1132%**。所有10轮均无新增autotune打印；仍存在轮间波动，不声称性能变成确定常数。这里是N64工程对照，不能直接当N256比赛分数。

640响应全部成功并逐项检查种子/用量/正文/TTFT/聚合；23冻结源和快照、2安装采样源、TopP重新渲染字节、实际83/131预热、Worker maxseq256/64与async、CV/KV16/S8及融合采样B64、流式命令差、KV容量和零preemption全本地审计。两边实际API进程39项非秘密profile环境已保存且逐项与screen映射后比较一致。

归档6328443bytes/25parts，SHA256 `e0741230d04ba7ce40ec7cad27937bcfca98ddbe4e45df7dec81925da59b68d5`。本机raw：`service-results/20261016_tianshu/combined_confirm_service/20261016_combined_confirm_service_tianshu`。复核：`python3 experiments/tianshu_combined_20261016/verify_downloaded.py --root <raw-root>`。

本轮独立收益刚过预设2%门槛。新组合Math和原官方脚本N256测试均尚未启动；公开赛题核查发现要求serve参数与基线一致，而当前控制及候选均包含额外调度参数。已向用户询问是否存在后续允许说明，后续采用前必须对齐官方启动命令或记录用户提供的最新规则。该核查同样影响此前4091.29成绩的适用范围：它是原官方benchmark脚本在自选服务参数下的已复核成绩，不能仅凭脚本未改即认定完整复现条件合规。

官方页面性能指标为total tokens/s，TTFT按公开baseline+1%作门槛，不是按我们优化后的最好TTFT作门槛。4K公开TTFT11573.53ms，按页面公式上限约11689.27ms；当前N64 TTFT不能与N256门槛直接比较。来源和公开JSON响应SHA见competition_command_review.json，规则链接：https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn。
