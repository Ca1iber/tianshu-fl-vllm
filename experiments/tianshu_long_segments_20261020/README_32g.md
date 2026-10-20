# K8V16长上下文分段：CPU准备阶段

当前K8V16的N>24解码固定4段；4K小测已比较过launch配置，但没有16K的相应组件覆盖。本试验让Graph捕获最大8/16/32段，再用GPU实际seq_len在8192及以下保留4个活跃段。只有producer和reducer各一个tiles-per-segment表达式改变；两侧除数仍是constexpr，不引入producer可变整数除法，参数ABI、KV格式和attention算术不变。

CPU已经验证两处文本可精确逆变换、其他函数AST不变，以及1..131072长度×三种最大段数的393216项覆盖。不据此声称GPU数值或提速。

准备9场景：B32/B40/B64的4K和16K，B40的17408/stride、8193边界与混合长短上下文ragged；控制staticS4、候选max8/16/32。每种同一Graph四相位变更Q/K/V/页表/长度，边界8193↔8186双向跨阈值。短上下文输出要求与S4逐位相同；所有相位完整GPU FP32/CPU抽查、完整writer存储、inactive segment哨兵、padding及RNG均需通过。9轮交错×20重放计入KV更新和单层attention。

当前正式16K独占GPU，本探针尚未启动。后续还需实际ModelRunner/Graph绑定、服务筛选和独立Math，不能把组件当新正式成绩。
