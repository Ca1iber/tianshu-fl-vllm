# FlagGems预热后DEBUG→INFO日志阈值服务对照

唯一差异：原始capture全部完成后，将flag_gems父logger级别DEBUG调为INFO。保持原FileHandler、once filter、传播设置及INFO/WARNING/ERROR/CRITICAL，保持全部算子、采样、RNG、权重与配置；启动的算子列表记录照常完成，稳态DEBUG不再构造LogRecord。不是关闭全部日志，不更改原生logging源码。

两组相同359来源合格模型父版本，62项实际环境、原13项serve命令、4096预算/maxseq64/async；明确不预存index_adjacent缓存、不接新采样预热。实际每步sample后核验有效日志级别，powers-of-two保存原handler/filter对象及7个算子child logger继承级别。原sample与capture参数和返回值直通。

纯CPU组件执行7个真实op源码日志位置/每轮13次调用及原LogOncePerLocationFilter AST，没有导入Torch/GPU库。control每两轮构造26条DEBUG、once实际写7条；candidate构造0条，全部INFO+记录与文本保持。CPU13调用局部119.723→7.222us，不计模型收益。

等单独index配置缓存对照终态后串行运行4K/1K C64 N64、3轮首轮剔除。有稳定服务收益才考虑独立质量和正式N256，不例行完整16K。截止北京时间2026-10-21 23:58。
