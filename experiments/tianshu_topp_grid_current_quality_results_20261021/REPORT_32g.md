# TopP16K/grid2：独立Math未通过，不采用

2026-10-21，BI-V150 32G，原版serve的13项命令，438运行源/62环境键/13独立控制器源。Math Level3完整105题，T1/top_p.95/max32768，并发8；全部请求成功，105份预测/评分的索引、usage、分数、原始行SHA和完整源码快照已独立重新核对。结果99/105=94.285714%，未满足严格>95%。有104题自然stop、1题达到32768上限；总输出342328 tokens。

完整原有INT8/权重/KV/图/采样检查通过，实际C8新采样路径已命中；43份权重与成功小测/此前合格版保持一致；prefill INT8实际M256、全42层路径通过检查。C8的Math不声称覆盖C64质量。

此前小测同次+0.83956%的服务吞吐收益保留，但不据此采用；正式N256四轮未启动，runner在质量门以退出码2停止。没有重跑本候选或挑选较好分数。不能只凭本次99与历史102断定全部差异由单一原因导致；本次官方质量门未通过已经足以不采用。

推荐仍为407源tianshu-original-command-4k-prefill-int8-20261021：正式4K5586.28 total tok/s、TTFT3780.63ms、Math102/105。ACCURACY_GATE的official_requirement_passed=false、QUALITY_ARCHIVE的strict_over95_passed=false明确保留；complete仅表示归档审计完成。

Git只保留本报告、105题分数/usage/hash和关键绑定JSON；完整原文/日志留容器/workspace/benchmark-results/20261021_topp_grid_current_math_tianshu。
