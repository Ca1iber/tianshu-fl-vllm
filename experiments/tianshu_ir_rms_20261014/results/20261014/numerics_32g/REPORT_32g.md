# 天数 BI-V150 32g：实际IR RMS provider语义验证170项通过

新provider tianshu_ir_rms直接注册vllm.ir.ops.rms_norm/fused_add_rms_norm，使用与native完全相同的参数名字、类型、默认值和返回schema。实现FP32累积/方差，归一化后先转weight dtype再相乘，返回activation原dtype；fused额外返回FP32相加后转activation dtype的residual。functional新分配输出，不修改输入；与旧Gems fused未中间转换的函数体分开实现，旧80项不复用。

170场景全部通过CPU FP32参考、当前实际IR native GPU参考（BF16 rtol=.016/atol=.016）、输入不变/输出非错误别名和Graph回放与同配置eager一致。143/170与native逐位相同，保留27项浮点差异。M1/7/64/128/576/2048、BF16和FP32权重、random/zero/cancellation/large/small，普通和fused两种调用；另覆盖N1024、FP32 activation、无weight、variance override1024、非连续输入回退。fused maybe_inplace验证同结果，初始实现保持native的functional不变输入语义。

120项实际选择新provider，50项选择native回退；24组真实compiled metadata/source记录匹配。guard为CUDA/BF16/2D连续N2048/M1..2048、连续权重BF16或FP32、variance_size=None或2048，fused要求residual同dtype/shape且连续。小M<=64用w8/s1，其他M用w4/s1。未改变globalEnable逻辑或数值检查门槛，supported用于静态CUDA可用性，动态supports_args仅检查参数。

实际IR源SHA d8e4380ca97d2c719836a7e15fb410a73d354b15d79fc54275b573bbb1c06910；provider实现源SHA 7b2b77bca09e5f83f722d65fe877252a9106ac7f02d24a8f612cc13d59224200。import hook在配置hash/worker priority初始化前注册，服务应使用--kernel-config.ir_op_priority=[tianshu_ir_rms,native]，避免平台native默认覆盖。micro仅用scoped set_priority，不能替代真实模型命中证明；正在独立短服务验证真实provider绑定，尚无模型吞吐/质量新成绩。

原始包2324385bytes/9分片，SHA256 f94404401e03c440dc462ae1a7975ca0af9814b0626dbbffc134e964cf4fc6bb；全部分片/整包、170唯一场景、24真实记录、3份执行源快照本地核验。原件service-results/20261014_tianshu/ir_rms_numerics/20261014_ir_rms_numerics_tianshu。

结论：数值与调用契约通过，可进行真实模型绑定及4K小规模对照。若端到端提升明确，仍需新独立Math105严格>95%及原脚本正式4K；当前旧103不可复用给新provider。正式最好仍3506.84/TTFT4494.12/Math103，tag tianshu-best-topp-w16-20261014。性能剖析：TODO，图内各kernel耗时未取得；IXsys仅证实整Graph覆盖。安装源/权重/原benchmark不变，沐曦只读。
