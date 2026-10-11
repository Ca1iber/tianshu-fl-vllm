# 天数FA2 Prefill tile实验计划（尚未启动）

源代码显示paged varlen Prefill默认mha_block_128使用BM128/BN32、4 warps、3 stages。计划验证实际已安装配置后，在独立进程只临时覆盖该heuristic，比较BN64/128及BM64/128等有限候选。用真实interleaved paged KV、GQA16/2、D128、BF16与causal prefix语义，不修改当前模型服务源码。

先进行固定FP32/CPU参考、padded tail和失败配置记录，再测预热后的交错时序。算子计时不作为服务吞吐收益；有明确服务收益后做独立105项数学和正式4K验证。当前64请求预算对照、后续质量任务优先，所有GPU任务串行。该文件只是待验证计划，没有声称tile优化已有收益。
