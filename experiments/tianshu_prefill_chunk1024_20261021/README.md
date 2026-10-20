# 4096总预算下限制每请求预填充块1024

保持合格LM模型精度、4096总预算、seq64/async、原13项serve参数；两臂都启用已验证RNG保持的采样预热。仅在原API默认预算、默认threshold0及匹配的Iluvatar MiniCPM配置下，把long_prefill_token_threshold从0改为1024，非零显式阈值、显式预算和其他模型等保留。显式传0与默认0无法区分，这里明确改变0值的默认行为，不声称显式0保持。

真实v1 Scheduler.schedule在运行和等待路径均应用该阈值。两臂具有同样的逐步观测：原schedule完成并更新num_computed_tokens后，通过before=after-scheduled识别预填充工作；候选逐请求断言工作≤1024，聚合直方图，检查真实阈值和预算。没有改request内容、采样或KV算法。

先做CPU真实EngineArgs AST、原budget/seq包装、边界/异常与observer测试，再串行小4K/1K C64/N64三轮剔首轮。实际GPU路径、数值、效果与新Math都尚未产生。CPU gate不代表服务或数学质量通过；源不在运行中改动。
