# 成对 K 读取的新一轮 MATH-500 验证

2026-10-20，使用成对 K 读取、warp=1 及现有 Future-KV 配置，重新运行官方 Level 3 全部 105 题。独立复核结果为 103/105，准确率 98.095238%，通过严格大于 95% 的要求。105 个请求均正常结束，无请求错误、无截断；总生成 353098 tokens，最长 32034 tokens。

本轮没有复用旧版本的质量结果。核验了每题评分、执行状态、token 用量、原始记录与整文件 SHA256；GitHub 仅保留评分、用量和校验信息，完整题目与回答仍在容器。

实际 C8 CUDA Graph 捕获和模型回放使用成对 K 源码 f1df731bae6cf48d5f38eef0a0adb5863ade2145667aab1ac976b9d473bdb002。记录包含 282 个运行源码、54 个环境变量及 8 个独立控制文件；独立审计确认正在进行的完整 4K 服务使用同一配置。

本轮质量已通过，正式性能仍待完整 4K 四轮结束。此前 N64 小样本的 5169.37 total tok/s、相对同规模对照 +2.5671% 不作为正式 N256 成绩。当前正式最佳仍以 TIANSHU_BEST.md 为准。

原始目录：/workspace/benchmark-results/20261020_pair_K_math_tianshu；正在运行的性能目录：/workspace/benchmark-results/20261020_pair_K_official4k_tianshu。本轮未安排完整 16K。
