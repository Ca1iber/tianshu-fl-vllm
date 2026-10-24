# O64 affine W4 C64新鲜质量：102/105，通过

2026-10-24，独立新API，全105题Math-500 Level3，C64、temperature1、top_p0.95、max_tokens32768。102/105=97.142857%，严格超过95%。105请求成功、0错误，104自然stop、1触及原输出上限，全部纳入分母；没有补跑、剔题或借用旧质量。错误index为18/87/101。

实际42层O64 W4捕获，QKV64仍42层BF16捕获；1836次真实FULL64图重放覆盖，不把Python路由调用当重放。959来源与快照SHA、65项实际环境、原13项serve参数保持；原GU融合、Down/LM W4及KV原生管理检查通过。C64使用FutureKV control以保证实际并发覆盖，其它精度路径与原r4相同。

新93585408字节buffer和42层native scratch全部纳入真实原生显存分析，模型7905652224字节、物理KV36479块；未声称净显存节省。完成后API/engine/evaluator终止、GPU空闲。此为C64补充质量，通过不等于已满足原C8质量；仍需新鲜C8及4K吞吐后决定是否采用。当前正式最好6050.94 total tok/s未变，也不把102分与旧100分差别当确定的准确率收益。

全原件留容器/workspace/benchmark-results/20261024_W4_O64_affine_math64_tianshu；关键逐题分数/stop、SHA与运行门槛见C64文件。原始归档由C64_RAW_ARCHIVE_32g.txt标识。代码、报告和关键结果进入天数GitHub。
