# BI-V150 32G：完整7槽骨干、2步Markov提案的实际服务验证

此前正确拒绝的7步配置小规模均值1660.33，对匹配控制2731.42无收益且仍调优。新原型保留checkpoint训练的7个非因果查询槽，仅对前N=2个hidden positions计算LM head与顺序Markov概率提案。anchor索引stride仍7；target只验证2个proposal+bonus，返回概率B×2×V，原始拒绝概率比与恢复核保留、正FP32阈值启用。

独立numerical先验证5个Q2/3/4目标及3个Q7双向draft场景、81CPU向量检查，通过。本次C8 actual serve采用原始模型、BF16、target FULL_DECODE_ONLY、draft eager，greedy和temperature1/top_p0.95两请求正常返回，各13输入、96输出token，greedy答案42，随机响应也正常生成。单题回应不代表Math正确率达标。

真实adapter record为query_slots7、sampled_prefix2、probabilistic/greedy各一条。Scheduler默认仅预留N+1=3槽，新过程局部hook至少预留7槽，目标验证宽度保持3；实际记录3→7。扩展原始DFlash按N+1创建的positions和slot_mapping缓冲区，B8容量均56，arange2105覆盖2048context+56query+1。避免7槽骨干跨页写入时只有3槽预留或buffer不足。

profiling/dummy按target N+1映射完整7槽query，LM head只预热prefix N，不额外draw RNG。没有使用未加载的confidence截断；仅调整固定合法proposal depth。GPU实际C8 smoke通过，C64/N128 4K/1K对照正在运行；尚无完整性能或独立105题准确率，未采用正式best。

完整smoke原始响应、log、Markov record、KV/buffer及冻结源码2045023 bytes，SHA256 4d8ef5d3894757573bbd055181abf90ab9abea9d06821822e39374e8cfc9fdd6，全部分块和整包本地核验。本提交为该可启动实验检查点。

质量/正式审计按实际N、固定7槽骨干、真实物理预留与buffer容量，要求冻结源和参数对应。只有新的清晰收益和独立Math严格>95%才原始N256四轮4K。正式最好仍2712.55 total tok/s、TTFT4720.38ms、Math102/105。

性能剖析：TODO；无新服务性能结论。未修改训练权重、安装源码、原始benchmark或沐曦仓库，未运行优化版完整16K。
