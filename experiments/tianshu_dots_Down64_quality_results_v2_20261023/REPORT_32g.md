# Down64 dotsaddz W4 v2：双并发完整质量验证通过

冻结模型998971f，同一版本新鲜Level3全部105题：C8 102/105=97.142857%，C64 103/105=98.095238%，均严格大于95%。两轮各104个stop及1个max_tokens（index83），全部计入105分母；C8错误18/87/83，C64错误18/87。无请求错误、无遗漏或补跑追分，不把随机采样差异解释成质量提升。

独立CPU重读全部210份review/prediction与官方report，核对原始文件SHA、checkpoint、916模型来源、85控制器及其来源快照，重算全部继承及新增profile/accuracy审计并逐字匹配。真实W4 M64 capture42，旧Down M64 BF16 capture0；42层payload、load warmup、实际worker memory-profile scratch和新增存储计入均通过。保持原op schema/fake，公开register_kernel只路由M64到新W4，其他M委托原backend；QKV/O64 BF16与LM64 W8A8保持。C64真实主模型M64图执行与新W4捕获链接通过。

QUALITY_GATE_32g.json记录完整门禁与原始文件SHA，KEY_SCORES_32g.csv记录两轮全部105题关键结果。raw位于/workspace/benchmark-results/20261023_dots_Down64_v2_math{8,64}_tianshu。大日志、答案、数组只留容器。

本结果仅证明本版达到质量门槛；整机吞吐待短4K同场对照，尚未采用，正式最佳仍r3 5950.75 total tok/s。
