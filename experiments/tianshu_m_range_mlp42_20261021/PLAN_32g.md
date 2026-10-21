# 42层真实权重 + 24个当前可达M + 完整MLP

候选固定BM64/BN64/BK128/W4/G8的two_dot融合，不改变已有W8量化和bank。覆盖当前已启用INT8的24个prefill M（128..2048 step128、2304..4096 step256），不扩大精度白名单。42份实际GateUp W8 NN、scale签名与原版逐一匹配，Down保留BF16实际权重，M2048保留当前special kernel；其他Down走实际vendor。对照使用已有unfused NN配置和从现用prefill操作生成代码提取的Inductor SiLU，保持原函数AST和decorators；独立重编译组件，不声称与服务二进制相同。

三相位动态A，全INT32和投影BF16位相同；实际timed激活.001+.02abs、完整MLP最终输出.02+.01abs，全元素检查并独立CPU重算4真实层的选点。所有42层实际bank参与每个M/相位，不用单层热缓存冒充42层。7组交替，每组3次42层Graph，返回每层时间。每层共享同fixture X而非上一层输出，不称完整transformer或服务；避免不含RMSNorm/Attention的递归MLP造成不真实分布。代码/门槛/源先冻结，GPU后修复另目录。只有GateUp完整192-case CPU gate通过且GPU空闲才启动。1小时或23:58北京硬停，之后仍需真实4K和新Math。
