# DSpark 七槽输入准备：160 项实际 GPU 调用通过

实际执行旧 DSparkProposer.set_inputs_first_pass、安装版 DFlash host 和原始 fused GPU input preparation；不加载模型、不执行 KV cache 写入/模型/采样/吞吐测试。10 种上下文 fixture、两次改变 past/bonus/page、四种目标提案数1/2/3/7、两种sample-view，共160调用全部通过。独立 CPU 再读全部保存 tensor，逐值重算 token、位置、context/query槽、seq_len、采样行。原始数据位于 `/workspace/benchmark-results/20261021_dspark_input_contract_GPU_tianshu`，冻结源及结果SHA见 GPU_GATE.json。

覆盖上下文1/3/17/256/257/513、多block网格、B8混合和B64、随机page且row stride带padding、被拒绝1～5个token、跨16-token页、接近131072最大长度；所有GPU query位置合法。CPU/GPU RNG、全部输入及输出边界guard保持不变，hidden对象身份不变。独立CPU复核与源码快照校验通过，未修改冻结源/CPU gate。

辅助层索引 [2,11,21,31,40] 与安装版实际辅助层方法一致。该输入准备路径没有发现导致4K接受率偏低的位置/采样行错误；这不构成整个草稿模型等价证明，尚未验证KV写入顺序、全部隐藏层计算或真实提案分布。

新 process-scope 七槽长度 policy 的 CPU 校验执行了安装版原 `_input_fits_in_drafter`，312个边界点含两种DSpark架构与普通DFlash对照。DSpark固定七槽时，旧num_spec_tokens+1对1/2/3提案会少计，对7提案会多计1；新policy统一按实际七槽，普通DFlash保留原规则。此policy尚未在GPU服务中安装，下一份接受率服务诊断才会验证实际hook；不会把输入组件当作服务已修复证据。

这轮没有新benchmark或Math成绩，不替换推荐版。后续小请求诊断比较自然文本与比赛随机token输入接受率，保持标准FP32 rejection和真实draft概率，再决定新的47层KV与Split-KV投机集成。
