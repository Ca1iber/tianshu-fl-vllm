# 原生FX融合三臂4K短测，2026-10-21

完整通过：control→candidate→reference，576/576个请求成功，零服务就绪后autotune，完整KV/attention/allocator/scheduler/权重/LM/prefill运行审核通过。所有源和快照由独立CPU重核。4K/1K、C64/N64，各3轮剔首轮；这是小规模对照，不替代N256正式成绩。

|版本|后两轮total tok/s|output tok/s|mean TTFT ms|吞吐spread|
|---|---:|---:|---:|---:|
|原始407源正式最好版本|5508.9690|1101.7938|10380.40|0.15652%|
|450源，静态compile64但不替换|5529.4693|1105.8939|10368.51|0.09642%|
|450源，静态compile64 + 42层双dot融合|5554.9744|1110.9949|10365.69|0.07679%|

候选相对真正原始407参考+0.83510%；静态compile64本身+0.37212%，融合相对静态控制再+0.46126%。候选超过预设整体+0.5%及spread<1%门槛，进入新鲜质量评测；单独静态版本未超过门槛。

35.94%是先前GateUp量化+FlagGems SiLU组件的速度提升，不能直接换算整模型；模型使用独立Inductor激活，控制函数与组件不同。现在模型吞吐已经直接测量，真实整体收益以上表为准。未证明模型任意浮点输入逐位相同，尚未采用、无新Math或N256正式成绩。

真实候选只在[64,64]图替换42个节点，43张动态图保持原样，M8 NN及24个prefill INT8暖形状/42实际请求路径保留。43个checkpoint/packed签名保持，未新增packed权重。首次接入的2条index调优失败保留原样；本新版本在启动实际预热输入63/64×索引32/4001，全部CPU值和RNG保持，计时请求期没有拦截index。

每轮完整分数见KEY_ROUNDS_32g.csv及ROUNDS_32g.json；冻结450源和真实原始407参考核验见SCREEN_GATE_32g.json。完整回复、日志与快照仅在容器 `/workspace/benchmark-results/20261021_nn_silu_fx_screen_{control,candidate,reference}_tianshu`。GitHub只保留代码、报告与关键结果。

正式最好仍5586.28 total tok/s、Math102/105。后续官方新鲜Math用batch8/max32768；另设计保持max32768的高并发Math覆盖实际M64融合，单独调整FutureKV准入以避免全额长请求预留把并发压到19左右。两项严格>95%后才跑正式N256四轮，不例行16K。
