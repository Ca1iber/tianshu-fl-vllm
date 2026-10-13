# BI-V150 32G：75点预热与固定索引组合独立Math通过

当前Native GQA L0 Prefill、Hybrid、adaptive Decode KV16/w4/stage1、预算2048，并开启75点Prefill分桶和固定last-token索引复制，使用官方Math-500 Level3完整105题、batch8、temperature1/top_p0.95/max_tokens32768、原始数据与chat endpoint。

本次独立评测102/105正确、97.142857%、105请求成功、0错误、105unique reviews均success，严格满足>95%。未复用昨天的102/105；错误索引18、87、101，与昨天最后一道错误不同，原始预测全部保留。

质量审计同时检查真实Native Prefill/Decode路由、75/75预热、int32 gather使用、模型runner/bucket/gather source SHA、主stage1与组合trace hook SHA及官方采样参数。这些开关和源码必须与正式吞吐对应，不能用buckets0/gather0的原评测替代新组合准确率。

原始全量2927369 bytes、12分块及整包SHA256 98ceaec2c577df29725f3f5e79e77a99fe14886e7896ceed437108bca1dc552a本地全部验证；预测、评审、报告各自SHA再次校验。保留真实评测文件及源码快照。

用户要求原始正式4K以及新增完整16K。完整4K N256/C64四轮排首轮已启动，之后完整16K N128/C64四轮排首轮，全部原benchmarkSHA不改、同新组合，GPU串行。当前完整正式最好仍2808.93/561.79 tok/s、TTFT4827.51ms，Math102；新正式成绩尚未完整完成，不以3270.51小测代替。CPU原型窄直方图计数仅准备，不进入本次质量或benchmark。

已补齐质量与正式启动器的buckets/gather/trace参数，并核对对应预算、源码和实际路由。--requested-benchmark仅响应用户直接要求而绕过短测收益筛选，所有质量、原始脚本、token/成功计数和源码守卫仍保留。原5轮控制与3轮候选只比较共同前三轮提示，正式流程自身仍四轮不变。新16K与4K独立目录与审计，保留每轮CSV及全量原始结果，未修改采样或权重。性能剖析TODO。
