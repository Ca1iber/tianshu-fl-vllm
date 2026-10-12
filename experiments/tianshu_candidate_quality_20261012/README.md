# 天数新候选独立质量评测

入口接受small-gemm、prefill-gemm或sampler单一候选，保持当前最佳Hybrid/adaptive/FA2tile128/budget2048，禁止通过该入口同时开启多项候选。官方Level3、105题、batch8、temperature1/top_p0.95/max_tokens32768，不复用任何旧正确率。

评测保存源快照与SHA、命令、环境、硬件、实际FA2与候选路由、全部预测和review。审计要求105个唯一预测与review、全部status success、无请求错误、GPU清理完成及实际候选路由内核SHA匹配。阈值严格>95%，即至少100/105。

服务稳定收益确认后才启动质量评测；仅有算子microbench收益不触发正式采用。对应质量通过后才考虑官方4K，没有优化版完整16K成绩。
