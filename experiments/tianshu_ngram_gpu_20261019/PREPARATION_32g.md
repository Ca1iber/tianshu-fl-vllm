# 原生 GPU ngram：下一项独立框架试验

源代码确认，CPU `method=ngram` 会关闭 async scheduling，原生 `ngram_gpu` 被
列入支持的异步类型；FL 固定版本的 ModelRunner V1 已实现对应 GPU 提案与
异步输入更新。本项使用原生 GPU proposer 和原生 target graph dispatch，
不安装之前的 248 点 mixed 策略，也不修改权重、target 上下文或采样参数。

沿用已验证的正值 FP32 标准拒绝采样、Q2/L0 attention 适配及固定 index；
整数 GPU proposer 算法保持原生，只增加不读取 GPU 值的 host 调用/shape
计数。hook 中明确要求实际 async scheduling 与 async spec decode 均开启。
服务前检查原48组拒绝采样、14阶段attention及48项index门槛与冻结源码。

独立小规模筛选为4K输入/1K输出，C64/N64三轮排除首轮，输入种子与当前
CPU ngram服务对照一致；保存完整响应、指标、实际环境及native图分支计数。
它切换了 proposer 和框架模式，不能把差值孤立归因于GPU算子或图覆盖。
目前未运行、未采用，没有新的Math-500或正式benchmark成绩。若出现可信的
收益，必须做新鲜质量验证和完整4K评测；不复用CPU混合图数值门槛作GPU
proposer的质量证明。
