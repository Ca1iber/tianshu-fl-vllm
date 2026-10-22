# LM动态批次补齐64服务候选：在原r3上单独优化LM

仅在冻结LM_pad64组件全部GPU/独立CPU核验通过、suggested_cutoff为32或48且所有对应已测M的eager速度比原实际路径快>5%之后生成选定计划并允许GPU。基线是原正式r3，保留原903源字节与QKV128/512/4096、O512/4096、Down/GU/LM8 BF16、KV/scheduler/sampling策略；不加入已被短测淘汰的QKV64上下文量化。

新LogitsProcessor._get_logits仅在实际M处于选定cutoff..64、BF16输入/无bias、logit scale=1、无softcap时使用已有LM NT INT8 pack与scale，动态量化后补零64、原固定M64整数GEMM、显式BF16舍入后FP32输出，返回前M行view；其他M沿用原始_get_logits，M<=8完全不加入新量化。M64算术与原LM逐位相同，其他M是新量化且必须本版本新完整105题C8/C64均>95%，不借用r3分数。所有q/scale/out/dummy均调用内分配，无新权重布局或payload、无全局张量workspace。

加载原LM bank后、实际DeviceMemoryProfiler内部做64零输入预热，核对原参数版本、全pack/scales签名、输入RNG、组件实际quant/GEMM CUBIN。profile_run在真实memory_profiling上下文再执行声明的zero64探针，记录峰值/临时字节和手工KV禁用；该显式探针不冒称真实请求。真实M范围与64调用由独立route记录核验，采样器不改，原13条serve argv和63环境变量。

旧dense_audit的LM64旧kernel计数守卫必须明确替换为新LM记录的实际M64整数调用和组件二进制/权重/显存核验；不能伪造旧binding。采用单独新版审计函数，只替换该LM64计数段，保留旧parent组合、42层GU图重放、LM8 BF16、全43bank、原参数/packing、KV/attention/scheduler/sampling/RNG/零抢占/无在线调优/环境/快照等其余检查。返回结果明确LM路由已更换，旧LM64执行不再声称。

新源/计划/选定范围/组件门槛/Gate全部在首个服务GPU前冻结，失败保留、修正另立版本。正确率后短4K N64三轮排首轮与原r3对照，只有>0.5%收益、热轮极差<1%、TTFT<=对照1.01才另建原N256四轮正式4K。GPU串行，今天23:58北京时间硬停止，不声称16K或完整赛事资格。
