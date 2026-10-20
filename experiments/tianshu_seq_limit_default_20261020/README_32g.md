# 原生max_num_seqs缺省256→64：仅CPU边界验证

本目录不是新的性能成果。10/14旧配置曾测max_num_seqs64，4K服务仅+0.6721%，未采用；今天已验证K8V16/S16和新暂存区候选的内存容量条件不同，先准备一个不改原始serve argv的缺省入口，是否值得GPU复测待新容量实验数据决定。

policy.py在原生EngineArgs._set_default_max_num_seqs_and_batched_tokens_args外层组合：先判断相同的天数/模型/布局/原始缺省条件，执行当前4096预算策略及原生默认一次，返回后仅在用户未设max_num_seqs时把缺省256改成64。显式max_num_seqs正值不覆盖，显式预算也不覆盖；其它模型、厂商、使用方式、并行模式不触发，异常原样传递。没有修改原生源码、scheduler、算子或RNG。

check_cpu.py执行当前安装版原生方法AST，覆盖选中/控制、显式64/256/32、显式8192预算、其它厂商/模型/API方式/并行，以及两模式原始异常对象和状态一致，共11项检查。native预算策略每次只调用一次。CPU_GATE_32g.json和原件20261020_seq_limit_default_CPU_tianshu保存证据；未占GPU、未跑服务/Math/正式benchmark、未采用。GPU验证仍需核验API和worker真实配置、graph捕获上限、真实KV容量以及完整请求，不能依据本CPU检查声称收益。
