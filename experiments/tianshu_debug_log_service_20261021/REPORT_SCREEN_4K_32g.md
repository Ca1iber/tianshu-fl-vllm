# FlagGems DEBUG→INFO：CPU日志成本下降，4K服务持平

2026-10-21，BI-V15032G，4K输入/1024输出，C64/N64、各3轮首轮剔除。原13项serve命令、同379份运行来源/62项实际环境，384条计时请求usage、43份权重签名、实际Graph/整数GEMM/小batch LM BF16、seed canary、零抢占及FutureKV释放独立复核通过。无新Math/N256/完整16K。

|FlagGems稳态日志级别|后两轮total tok/s|TTFT ms|极差/均值|后两轮调优|KV块|
|---|---:|---:|---:|---:|---:|
|DEBUG原始10|5421.118977|10910.743449|0.161352%|0|40731|
|INFO候选20|5417.064502|10870.436531|0.116946%|0|40723|

吞吐−0.074790%，TTFT−0.369424%，变化处于本次波动范围，没有明确模型收益，不采用、不继续新Math或正式N256。CPU13条真实DEBUG调用119.723→7.222us是局部改善，不能直接当模型提升，也没有时间线证据推断它被CPU/GPU重叠隐藏。

原始控制三轮5226.637535、5425.492522、5416.745432；候选5229.703877、5413.896988、5420.232015，全部计时期间调优0。候选在原capture完成后才调整阈值，启动算子列表照常记录；实际最后powers-of-two快照至少2048次sample均保持INFO，7个相关op子logger继承20，原handler/filter对象、传播设置与INFO/WARNING/ERROR/CRITICAL保留。该2048是已保存观察下界，不是整个服务最终sample总数。

不预存index_adjacent配置、不接新采样预热，不改模型/算子/RNG。KV块相差8按实际记录，不宣称两组容量完全相同，也不据此归因。完整响应、文本、日志和源码快照留容器`/workspace/benchmark-results/20261021_debug_log_service_{control,candidate}_tianshu`；关键结果与原件SHA见SCREEN_GATE/ROUNDS/CSV。
