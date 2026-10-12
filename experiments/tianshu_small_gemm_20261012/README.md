# 天数M8条件GEMM服务实验

仅针对已完成120/120 CPU选定点与完整GPU FP32数值检查的M8三个形状：qkv(N2560/K2048)和o_proj(N2048/K2048)用BM16/BN64/BK64、split2；down(N2048/K6144)用同tile、split4；warps4/stages2。gate_up、M64及其他形状保持原vendor实现。原micro测量分别约−45.20%/−45.56%/−25.76%，尚未证明服务收益。

进程级hook要求vendor mm源SHA4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090、数值验证kernel SHA86a2244f3fb30a8294d53174bd3aa6080f947b5b80ce43b6212a35b533463d56。只处理BF16、连续a/c、NT连续权重；直接把native预分配c交给相同验证kernel，不重排权重，不增加输出copy，FP32 partial单独分配。通用源码未改，只在显式实验环境替换vendor launcher。

当前C8/N16、4K/1K、三轮排首轮服务对照正在运行，保持Hybrid1/adaptive1/tile128/budget2048。需要三种实际路由记录、原始响应和源快照。新配置独立数学和C64正式成绩均未完成；若小负载收益明确，再设计对应标准负载和质量验证。当前正式最佳仍tianshu-best-fa2tile128-20261012。
