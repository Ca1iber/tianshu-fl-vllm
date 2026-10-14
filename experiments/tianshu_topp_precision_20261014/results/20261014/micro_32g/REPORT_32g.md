# 天数BI-V150 32g：TopP精度修复与16K tile独立试验

本地算子试验，不是官方吞吐成绩；当前正式最佳4091.29 total tok/s不变。对当前原生TopP做组合精度修复：概率域精确重复值计数、更严格搜索停止范围和迭代上限、常量输入搜索区间修复、在概率域比较最终阈值。原TopK分支AST完全不变，TopK实际回退原生内核。不能把组合修复收益归因单项机制。

六配置、十种输入，含BF16往返normal/peaked、相等值、masked、ties及TopK；每个均检查原输入与翻转输入、CPU FP64参考与1e-6质量门槛、Graph输出等于直接执行。四个修复配置全部严格通过。原生基线语义误差保留，不因与基线相同而豁免修复候选。

|输入|原生ms|修复16K/w16 ms|耗时变化|
|---|---:|---:|---:|
|B8 normal topk=False|1.150198|0.873681|-24.041%|
|B32 normal topk=False|2.221877|1.747950|-21.330%|
|B64 normal topk=False|4.580882|3.514594|-23.277%|
|B64 peaked topk=False|2.807751|2.284309|-18.643%|
|B8 masked topk=False|1.151756|0.852842|-25.953%|
|B8 ties topk=False|0.765739|0.587529|-23.273%|
|B32 normal topk=True|0.556091|0.554942|-0.207%|
|B64 bf16_normal topk=False|4.202108|3.062815|-27.112%|
|B64 bf16_peaked topk=False|3.161469|2.307152|-27.023%|
|B8 all_equal topk=False|0.272305|0.370320|+35.994%|

all_equal修复后更慢；TopK均原生回退，没有声称加速。定时包含相同输入还原copy，420原始events复算，4224行语义检查复算，四冻结源码/native源码/生成源码及kernel artifacts SHA全部本地复核。未采集完整模型profile；独立算子耗时不能换算模型吞吐。

下一步：与当前采用配置做新启动服务对照，质量和正式成绩必须重新验证。

原始记录：service-results/20261014_tianshu/topp_precision/20261014_topp_precision_tianshu/micro；归档信息见ARCHIVE_MANIFEST_32g.json。
