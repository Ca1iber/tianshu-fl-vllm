# 中性单位温度：同次短4K筛选 +0.646%，尚未采用

2026-10-23，BI-V15032G。父项为已发布r4：原944来源、原13项serve命令、同一模型/权重/精度/Graph/KV/调度。实际InputBatch CPU请求全精确T=1且all_random时，采样ContextVar确认metadata及temperature tensor身份后，candidate仅跳过FP32的除以1；其他情况完整回退。control做相同观察/判断并执行原除法。原参数、p与RNG不改，无GPU数值读回。定期记录每128次调用，双方等量。

前置8组真实Sampler：B1/B8/B64、T=1/.7/1.4/混合、默认/部分/全部逐请求generator，完整温度缩放bits、最终token、global与逐请求RNG状态相同。初次组件同名module导入错误与初次控制观察计数未落盘错误均原件保留；后者audit_failed未改为通过，v2仅修日志且用新控制/候选完整重做，计算AST一致。

|方案|首轮tok/s（剔除）|第二轮|第三轮|后两轮均值|TTFT ms|后两轮波动|
|---|---:|---:|---:|---:|---:|---:|
|同次r4 control|5557.80|5881.11|5946.04|5913.58|8608.49|1.0980%|
|跳过实际T=1缩放|5578.63|5929.82|5973.74|5951.78|8586.02|0.7379%|

3轮N64/4K输入/1K输出、并发64，双方384个计时请求的用量和成功状态逐项核对。均值吞吐+0.646111%，TTFT-0.260961%；第二、三轮同序分别正向，但仅两轮热态，控制自身波动约1.10%，仍需后续复测。不得把组件除法成本或这一小幅短测结果当作正式N256提升。

独立CPU核对全来源与捕获快照、原944源、实际65个环境键与13argv，归一化路径后环境唯一差异为mode。4个冷salt seeded canary全文相同；热态新增GEMM autotune=0、抢占=0；实际C64 control保留原除法、candidate跳过记录均>=1000。双方GPU均正常清空。见KEY_COMPARISON与全轮KEY_ROUNDS。

仅性能筛选审计通过：没有宣称已过全部继承正式审计，没有新105题Math、N256正式4K或16K。此版尚未采用、未替换r4。当前正式最好仍6050.94 total tok/s、1210.19 output tok/s、TTFT3076.66ms；r4 Math C8/C64各100/105(95.238%)。剩余时段不足完整正式验证，本日结束GPU工作并保存全部证据。

完整响应、日志、metrics和source_snapshot仅容器 `/workspace/benchmark-results/20261023_unit_temperature_screen_v2_{control,candidate}_tianshu`；GitHub只代码、报告、关键JSON/CSV。下一次优先复测此小幅收益与正式资格，模型内部无图GPU诊断仅准备尚未执行，不能当已有热点结论。
