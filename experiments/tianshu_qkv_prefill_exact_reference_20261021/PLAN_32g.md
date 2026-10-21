# 在质量测试前直接对照未包装的原407版

QKV V2两臂共用新的opaque custom-op，control数值继续调用原apply，但这只证明方法委托，不能单凭该记录声称原编译上下文、实际GEMM kernel或逐位BF16输出不变。V2 control稳定5481.31，历史其他407小测约5507..5517；这些跨时间差异不能独立归因，但要求候选不能仅靠包装后的较弱control取得表面收益。

本计划在V2候选臂开始、尚无候选计分结果时声明。只有原先V2 SCREEN预声明门槛通过，才额外调用完全未改的tianshu_prefill_int8_service_20261021/run_service_trial.py，mode=candidate，即当前407源推荐配置，新的独立N64/C64/4K/1K三轮、排首轮。所有407 runtime源必须是V2 432源的完全相同子集；62env仅允许output root与明确PYTHONPATH hook目录差别，原13serve参数相同。该实验不改V2冻结代码或原门槛。

候选相对原407的增幅还必须>.5%且超过两臂极差，各自spread<.5%、TTFT退化<=1%、排首轮autotune0、usage全额/零抢占/FutureKV释放和原审计全部通过，才值得新Math。仍需105题>95%，最后正式N256证明比5586.28快。参考臂不能使用V2新wrapper；不常规16K。23:58北京GPU硬停。源码与关键结果GitHub；无本地副本。
