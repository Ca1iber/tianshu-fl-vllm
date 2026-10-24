# O64 affine W4 原C8新鲜质量：102/105，通过

2026-10-24，全新API，Math-500 Level3全部105题，原C8、temperature1、top_p0.95、max_tokens32768、timeout3600、ignore-errors。102/105=97.142857%，105请求成功、0错误，所有输出上限结束也纳入分母。没有补跑、剔题或复用旧预测。原件均由原始controller完成运行和审计。

与已通过的C64补充质量同一959来源、65项实际环境、原13项serve参数；两组均102/105，满足本版质量门槛，不将相对旧版本100分差异当确定准确率改善。C8本身并不代表已覆盖O64运行路径；该路径由C64的42层真实W4捕获及1836次实际FULL64图重放单独证明。当前C8实际覆盖计数、逐题分数/stop及原件SHA见C8 JSON/CSV。

所有新增buffer/native scratch已通过原生加载与峰值显存检查，原BF16/W8保留，KV容量按实际分析分配。此质量通过后，已启动同payload的4K N64四轮控制/候选对照，预先排除首轮；尚未产生新的吞吐结论或正式成绩，当前最好6050.94 total tok/s不变。完整质量原件与controller结果留容器/workspace/benchmark-results/20261024_W4_O64_affine_math8_tianshu。GPU计时结束后再执行归档和批量源核验；本次只在控制组启动阶段保存小型关键结果。
