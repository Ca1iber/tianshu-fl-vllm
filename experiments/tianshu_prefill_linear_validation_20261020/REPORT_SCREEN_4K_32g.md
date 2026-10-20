# 天数 GateUp BF16 linear：真实服务小规模对照

42 层 GateUp 的大批次 BF16 fallback 在 native profile/预热完成后，candidate 对 M>=128 使用 `flag_gems.linear(x.contiguous(), weight, None)`；control 保留 qualified mm。两组都保留 M8/M64 真 INT8、全部 43 组原始权重/量化 payload、GPU4 加载、同一 KV/调度/采样策略和原始 serve argv。

| 模式 | 第一轮 total tok/s，排除 | 第二轮 | 第三轮 | 后两轮均值 | 后两轮平均 TTFT ms |
|---|---:|---:|---:|---:|---:|
| qualified mm | 5117.78 | 5294.94 | 5309.57 | 5302.25 | 11551.36 |
| 大批次 BF16 linear | 5230.58 | 5426.74 | 5419.01 | 5422.87 | 10890.77 |

均为 4K/1K C64 N64，三轮每轮 64 成功请求。后两轮吞吐提升 **2.2749%**，TTFT 约减少 **5.72%**。两组首轮后均无 Triton 现场 autotune，波动低于 1%。完整 348 项相同运行时源文件及 snapshot、61 项实际环境已独立复核；两组全部 43 bank 原始 BF16/INT8/scale 哈希相同，并与既有合格 Math 一致。

路由在 ModelRunnerFL.capture_model 入口启用，并验证此前 42 层 M2048 原生预热已完成。实际调用记录区分 before_capture/capturing/ready 与是否在 CUDA Graph 捕获内；已核验 ready 后真实请求全部 42 层命中候选 linear，control 则命中 mm。M8/M64 INT8 没有被该 fallback 拦截，C8 与 C64 的捕获/实际重放保持有效。

此处是独立服务小规模实验，不是正式 N256 成绩。尽管权重 payload 相同，BF16 GEMM 实现发生改变，必须重新跑官方 Level3 的 105 道 Math 且严格 >95%，并核验 Math 中全部 42 层实际使用新路径；不能复用旧准确率。质量通过后才跑正式原始 4K 四轮，未通过则停止验证链。报告写入时新 Math 已开始、结果未定。

既有正式最好仍为 5375.50 total tok/s、4191.14 ms、Math 100/105。完整 16K 未排入本轮。

原始结果在容器 `/workspace/benchmark-results/20261020_prefill_linear_service_{control,candidate}_tianshu`。GitHub 仅保留代码、SCREEN_GATE/SCREEN_ROUNDS 和本报告。
