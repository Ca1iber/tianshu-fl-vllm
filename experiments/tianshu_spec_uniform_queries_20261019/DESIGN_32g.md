# 下一项待验证：请求内补齐 query，复用 uniform Q2 图

这是设计草案，尚未实现、没有 GPU 数值或性能结果，不改变当前服务对照。

保持真实请求数 N，只对每个 Q1 请求在真实 token 之前插入一个不计分的 dummy query，使每个请求的物理 query 长度都为 2，总行数 2N；Q2 请求保持两个真实 query。真实 hidden 输出再映射回原 compact 顺序，供原 logits gather 和标准拒绝采样使用。所有真实请求、真实 logits 和采样仍保留。

Q1 的 dummy position=L-2、真实 position=L-1，seq_len 仍为 L；dummy slot=-1 不写 KV，真实 slot 不变。Q2 两行仍为 L-2/L-1。因 dummy 放在真实行前，真实 Q1 的 causal 终点仍为 L-1；插在后面会改变真实 causal 终点，不可采用。仅对已完成 prefill、L>=2 的 decoder 验证批次考虑此方案。

BatchDescriptor(2N,N,uniform=True) 与原 num_spec_tokens=1 的 uniform 图语义一致。优先复用已有键，只为 N=16..64 中缺失的精确请求数添加 uniform 键，不补齐请求数，不使用原 mixed 策略的 248 个键。原 dummy_run 已能产生 all-Q2，无需修改其元数据算法。prefill、LoRA、encoder、cascade、DP、ubatch 以及不支持范围均保留原路径。

需要 GPU 常驻小工作区保存原 compact input_ids/positions/cu/slot；真实 replay 前 pack 到模型原固定输入缓冲区，模型后按 real-row map gather hidden，并恢复原 metadata/input 缓冲区，避免影响 sampler、proposer 和后续步骤。capture 与 runtime 必须使用同一输入地址；不可仅替换为新分配张量而复用旧图。

验证顺序：CPU 穷举 Q1/Q2 分布，核验输入/输出映射、位置、slot、causal 终点、native uniform key 与异常恢复；独立 attention FP32/CPU/变输入 Graph 校验；原编译真实模型完整 hidden/logits 对未补齐执行校验、42层 KV 恢复、RNG；随后才做无回调服务 A/B。dummy 行不会进入 logits 或采样。不能拿前一项 token 尾部补齐的八样本门槛替代本方案门槛。

主要代价：当 draft 提案少时 token 计算最多增加一倍，额外 pack/gather 也可能抵消图重放收益。需要根据正在运行的 248 点精确请求数服务结果决定是否投入实现，不能预设会提速。
