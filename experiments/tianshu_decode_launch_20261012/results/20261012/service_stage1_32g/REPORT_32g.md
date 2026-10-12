# BI-V150 32G：保持KV16改stage1端到端收益

已验证native GQA L0Prefill +Hybrid +adaptive Decode，budget2048；只将3D unquant/Q16-KV2/head128/B<=64的TILE16/w4/stage3改stage1，kernel body、grid、分块和数学公式保留。长8场景与14短KV/填充×stage1/2数值检查通过。

4K/1K、C64/N128三轮：1526.297791、3101.832135、3271.482110 total tok/s，各128请求成功；384usage重新核对4096输入/1024输出。排首轮均值3186.657122，对native1同规格控制2742.829558提升16.181376%。两测量轮区间完全高于控制，但极差/均值5.323760%，仍需原始正式协议确认。三轮均0新增调优；首轮低速不能直接归因调优。

TTFT7877.185953→7568.634795ms，略改善。真实模型记录64类batch路由，compiled4warps/stage1、原kernel SHA bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0，native Prefill原hook和生成L0 kernel保留。无模型权重/安装源码/数据/benchmark修改。

这是N128小规模三轮，不是原始N256四轮官方分数。新的独立105题Math已启动，严格>95%后才原脚本四轮4K，未借用102/105旧成绩。正式最好仍2712.55、TTFT4720.38、Math102/105，当前候选尚未采纳或打best标签。

完整3172108 bytes、13分块及整包SHA256 0c86674df11ec9c66fd77f4dc4382dcbdbffab21a52bf4b4be98791895cf2419本地校验，保留384响应、rare probes、metrics/log、原生Prefill与decode路由、源码快照。源和审计随Git记录。性能剖析TODO；未跑优化版完整16K。
