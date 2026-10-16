# 天数完整Sampler CUDA Graph探针

CPU诊断已确认C64 logits为BF16、纯随机TopP与中立内置processor。该实验在原Sampler.forward阶段捕获原始完整采样，包含原BF16→FP32、TopP、softmax、指数RNG及int32输出；不前移采样或改变算法。每次复制logits/temperature/top_p到自有静态缓冲，返回独立输出以适配异步调度。

仅C64/130560词表/BF16连续输入、纯TopP、无额外约束/seeded generator/logprobs/TopK时尝试。未知processor、活跃MinP/MinTokens/LogitBias、thinking holder等全部回退。预热与捕获必须保持默认RNG状态，并在实际后端通过与eager的完整序列及状态相等检查。

V1实际失败：warmup/capture恢复默认RNG状态通过，但首个变输入case的64个token与eager不同，即使最终RNG状态相同。失败原件保留，不接入服务。实际FlagGems指数采样在Python取得seed/offset后以标量传给kernel，需单独处理捕获时随机参数的生命周期。

V2保留原指数采样在Graph外每次真实执行一次，仅将其生成的新noise交给Graph内原始softmax/divide/argmax；仍采用原随机算法、形状、dtype及offset，捕获前预热的RNG状态恢复。实际9组576token与8步混合512token、RNG状态、动态输入及旧输出生命周期均通过，但完整staged调用3.491334→3.644673ms（+4.392009%耗时）。不接入服务，不跑新Math或正式benchmark，见[报告](results/20261016/micro_32g/REPORT_32g.md)。
