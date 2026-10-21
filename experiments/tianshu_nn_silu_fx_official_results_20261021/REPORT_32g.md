# 天数正式4K：静态M64与NN GateUp/SiLU融合

原始serve命令，4K输入/1K输出/C64/N256，四轮排首轮。最终5618.01 total tok/s、1123.60 output tok/s，平均TTFT3835.46ms。相对上一正式5586.28，吞吐+0.5680%；TTFT从3780.63增加1.4503%。后三轮吞吐最大差距0.08544%，1024/1024请求成功、0抢占、没有就绪后的现场调优。没有运行本版16K。

|轮次|total tok/s|output tok/s|TTFT ms|耗时s|
|---:|---:|---:|---:|---:|
|1（预定排除）|5451.72|1090.34|5549.84|240.42|
|2|5615.61|1123.12|3757.04|233.41|
|3|5620.41|1124.08|3911.58|233.21|
|4|5618.02|1123.60|3837.75|233.31|

相对公开基线2028.01 total tok/s，提升约177.02%。这是本次三轮均值与历史/公开值比较，非同轮正式A/B；此前小测三臂实际对照为+0.8351%。TTFT略退，故保留上一版标签供比较，按总吞吐记录本次新高。

官方Math8新鲜101/105=96.19%，补充C64新鲜103/105=98.10%，均严格>95%。两份105题均全部成功，模型同450项数值源。补充只将FutureKV准入candidate→control以使64请求真实覆盖融合，原生allocator保留；正式benchmark恢复canonical candidate。13项serve参数、62项环境、原407源的精确子集、42层M64真实融合图重放、M8/动态图/权重/LM/prefill/attention/KV等全部原审核通过。

原controller完成GPU后在CPU审核因缺少state.workload而audit_failed，原result保留原样。独立CPU恢复只从未改的benchmark命令及四行CSV推导workload.input_tokens=4096等字段，随后调用原版完整validators，包含zero_preemptions和真实64/4K图帧检查，全部通过；未改模型、原审核代码/条件、原始数据，未重跑GPU。恢复审核器还证明全部旧审核依赖字节等于GPU前Git21f9a0e。完整CPU恢复证据在../tianshu_nn_silu_fx_official_recovery_20261021/OFFICIAL_RECOVERY_GATE_32g.json，完整逐层记录仅容器并有SHA。

配置与数值路径见已冻结tianshu_nn_silu_fx_screen_20261021/candidate_profile.py。静态compile64加42个NN INT8 GateUp/SiLU融合，原模型/原BF16权重保留、43个INT8 bank总1326938112bytes没有额外packed payload。启动四项真实index预热，稀有index不在请求期调优。新部署入口仅调用同一450-source profile，不更改官方serve参数。

用户最新M范围方向继续：QKV/O/Down布局首个组件已准备24M×2phase，当前NN prefill本就有24个INT8 warm点；之后分开补覆盖、GEMM调度和激活融合，按各角色实测区间筛选，不把沐曦五点直接推广全区间。原始大日志/张量留容器，GitHub保留代码、报告和关键评测。
