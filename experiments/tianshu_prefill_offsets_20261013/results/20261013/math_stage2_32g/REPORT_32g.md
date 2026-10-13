# 天数32g：新Prefill stage2+Decode32独立Math

重新运行官方Level3 105题，新配置103/105、98.095238%，0执行errors，105预测/评阅索引唯一且完整，严格>95%。错题索引18、87。未复用旧stage1配置的102/105。本次temperature1.0/top_p.95/max32768/batch8和timeout3600与官方一致；单次多正确一题不能据此断言准确率提升。

实际Prefill M256/KV128/w16/stage2配置与路由都存在，和正常16K/1K小测+2.425%的overrideSHA一致；Decode32 hook/helper、生成c407、Native L0 float body4a1、B75G1和budget2048其余源相同。新stage2源、实际env、route、configuration、generatedHash保留在accuracy_audit，不能只看旧auditor模板stage1基础字段。没有混入SQL或KV写入实验。

全部105预测/判分/日志/源快照归档3081325bytes、12分片；SHA256 dce2352578f3ba6aee04e41bff9275e45780673ed217d6f343df770dd367e80e及各分片全部本地verified。目录service-results/20261013_tianshu/prefill_stage2_math/math。run_stage2_math.py逐字匹配执行快照。

质量通过后，顺序控制器开始原始16K/1K C64 N128四轮排首轮，再原始4K/1K C64 N256；正式stage2成绩尚未闭合，旧已验证配置继续保留。只因服务+2.425%且计分轮不重叠才追加原始正式复测，benchmark源未修改。
