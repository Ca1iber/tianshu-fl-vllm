# 请求索引 int32 与位置 gather 融合实验

范围：保持 positions/query_pos 的 int64；只将有界请求行号缓冲改为 int32，并单独试验融合 gather+cast+add。原 KV slot mapping、调度规则、采样与 RNG 均保留。融合仅用于无投机、普通位置编码路径，其余使用原表达式。

先运行真实 `_prepare_inputs` 方法的控制缓冲实验，覆盖纯 decode、混合 prefill、动态稀有 M、256 请求边界及 Graph 输入变更。CPU 整数 oracle 检查位置；slot mapping 在此是参数观察桩，不能代替服务验证。冷调用包含已有本机缓存和首次 JIT 的混合成本，不代表干净缓存启动；稳态事件/主机计时单独记录。达到收益门槛后才运行独立服务对照，不把组件耗时当正式吞吐。
