# 原生GPU ngram：异步已开启，服务仍慢于当前最佳

2026-10-19，BI-V150，执行源码c40226a。官方主serve参数保持，额外speculative_config为ngram_gpu/k1/min3/max5；未安装248点mixed策略，target图调度保持原生。实际进程验证async scheduling与async spec decode都为True。

|轮|Total tok/s|Output tok/s|Mean TTFT ms|新增调优|
|---:|---:|---:|---:|---:|
|1|985.060|197.012|144563.50|216|
|2|2380.237|476.047|15974.01|12|
|3|2635.074|527.015|15965.00|4|

排除首轮后均值2507.656 total tok/s，仍不采用。这是4K/1K C64/N64筛选，不是正式N256成绩。GPU路径切换了整个proposer/异步框架组合，不能孤立归因于单个kernel或据条件接受率解释全部差值。

首轮216条mm_kernel临时调优，后续12/4条，prefill75保护仍关闭，因此没有解决稀有GEMM稳定性。异步开关恢复并不自动构成收益。下一步独立计时原生已编译GPU搜索，评估两阶段整数搜索是否有实际成本依据，再决定是否接入服务；另需关注prefill稀有M。

本地独立下载审计通过全部192响应/5个固定种子canary、93份源码快照、42项实际环境、8份metrics及native host图分支计数。维持标准正FP32拒绝采样，无synthetic acceptance或FP64噪声。没有新Math-500，没有新正式benchmark，不借用当前最好配置的质量分数。

原始归档5,994,393字节，SHA-256 `b278d8904d273c4084696515fb0685cb1a9a6fb803afc47290b18e0b0eeb21f5`。本地目录service-results/20261019_tianshu/ngram_gpu_service/20261019_ngram_gpu_service_tianshu。正式最好4093.81 total tok/s / Math102/105保持。

额外覆盖证据：三轮共3015次decode/verify全部为原生NONE/eager，实际FULL replay为0（prefill另399次NONE）。其中2804次混合Q1/Q2、211次全Q1，force_uniform均None、force_eager=False。异步开启并未使这些ragged批次自动满足uniform=2捕获条件。此结论来自已封存实际host分支计数，不是从启动捕获50点推断运行覆盖。
