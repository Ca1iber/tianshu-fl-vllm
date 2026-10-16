# 天数完整Sampler CUDA Graph探针

CPU诊断已确认C64 logits为BF16、纯随机TopP与中立内置processor。该实验在原Sampler.forward阶段捕获原始完整采样，包含原BF16→FP32、TopP、softmax、指数RNG及int32输出；不前移采样或改变算法。每次复制logits/temperature/top_p到自有静态缓冲，返回独立输出以适配异步调度。

仅C64/130560词表/BF16连续输入、纯TopP、无额外约束/seeded generator/logprobs/TopK时尝试。未知processor、活跃MinP/MinTokens/LogitBias、thinking holder等全部回退。预热与捕获必须保持默认RNG状态，并在实际后端通过与eager的完整序列及状态相等检查。

当前仅准备探针，尚无GPU能力或收益结论。`probe_graph.py --output NEW_REMOTE_ROOT` 独立无模型验证动态输入、温度/p变化、同RNG token/state、混合eager/Graph、旧输出生命周期和回退边界，之后再测包含输入/输出copy的完整微调用耗时。能力失败保留失败原件，不放宽语义断言。
