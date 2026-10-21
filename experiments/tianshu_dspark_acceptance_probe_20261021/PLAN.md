# DSpark 接受率短请求诊断

不属于新性能/准确率成绩，不比较整模型吞吐，不替换5586.28推荐版。使用旧DSpark BF16标准probabilistic+FP32 rejection兼容路径，目标FULL_DECODE_ONLY、draft eager、C8、预算2048、maxlen131072，保持固定七个训练query槽且只提案两步。唯一新增功能为已通过312 CPU边界点的七槽最大长度检查，实际request hook必须记录42+5层、共享BF16 LM head、辅助层[2,11,21,31,40]、标准probabilistic配置。旧0.5944是C64长输出历史值，不作为本轮对照吞吐。

CPU准备8条真实Level3题目的chat-template tokens；随机匹配组每条输入长度精确相同。另有8条4096随机token，采样区间沿旧小实验[1,119999]，不声称与官方random dataset生成器逐位相同。三组先各16输出token暖身；再各两轮128输出、T1/top_p.95/ignore_eos、C8并发；两轮顺序ABC/CBA，同round各组每题seed相同。全过程9组72请求。Math只是自然文本来源，128token输出不算105题评测或准确率。

请求前后等引擎running/waiting归零后读取Prometheus counters，精确计算draft/accepted/per-position增量，校验两步提案数关系。保存全部响应/usage/metric/随机输入，逐组报告接受率；计时仅诊断，autotune日志完整保留，不拿冷启动或短输出当吞吐成绩。每组usage必须与输入长度/输出数一致。标准rejection kernel不改，真实dense draft probabilities不改。

源清单是明确列出的primary hooks/proposer/runner/attention/rejection与CPU证据/提示数据，不是假称全Python传递依赖闭包。所有清单源和CPU_GATE在GPU启动前冻结并快照；启动后不改任何py/sh或CPU gate，真正修复新目录，元数据恢复单独CPU目录。GPU最多40分钟、客户端最多15分钟，并受北京时间23:58硬截止；Bash trap只清理本进程组，不杀其他任务。所有源码/原始数据仅容器，GitHub只代码/报告/关键JSON。
