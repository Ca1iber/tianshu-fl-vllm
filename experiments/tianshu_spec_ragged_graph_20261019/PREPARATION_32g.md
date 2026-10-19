# 天数混合投机批次独立图候选：CPU准备检查点

基于已独立核验的真实936次mixed/73次Q1/no FULL replay诊断，源码级候选提供42个独立uniform=False描述符，ReqCapacity=16/24/32/40/48/56/64，每组QueryCapacity从Req到2Req按8递增；仅实际Req16..64、maxQ2且混合1/2、prompt已完成、无cascade/encoder/DP/ubatch、原runtime为NONE时选择。query padding限制≤12.5%；其他路径调用原实现，原全Q2图保持。

CPU_POLICY_AUDIT保存真实原生CudagraphDispatcher和BatchDescriptor AST执行，而非只用自写stub替代关键规则。42种实际生成dummy元数据前缀执行通过；2144种N与Q1/Q2组合中1741路由、403回退；6个force_eager/force_uniform/cascade/encoder/未完成prompt/缺图边界回退；两种短prefill/全Q1不误匹配候选键，原warm返回和异常及finally恢复通过。方法派生仅改变dummy max_query_len和query列表，不修改模型/attention浮点主体。Req==Query容量的捕获样本含一个Q2和一个Q0，保持总query/Req正确；实际运行额外Req的cu/seq_lens和slot=-1仍须GPU验证。

自有捕获描述符通过get_capture_descs加入同一FULL组，以保留显存估算按mode分组的前提；不加入原生dispatcher的key set。后一项避免(tokens==requests)混合键误匹配原生的全Q1或短prefill路径。实际路由还要求对应GraphWrapper entry已有图，禁止运行时临时捕获。

CPU重放已审计诊断分布：可覆盖801/936混合调用（85.5769%），额外query行2539/54773=4.6355%，明细COVERAGE_PLAN。仅Req64原型覆盖333次，已据分布扩展；这是容量推导，不是性能结果。

尚未部署本候选，尚无GPU/model结果。必须先接独立hook和真实模型数值验证：原参数+同ngram/fixed-index/正FP32 rejection，至少8个改变输入/QL组合，覆盖Req与query补齐、ReqCapacity==QueryCapacity；对比同补齐eager与完全unpadded eager的实际hidden/logits。双次参考前向写KV后必须恢复候选触及的KV页，保证后续请求状态属于候选；额外检查未触及守卫页、slot=-1、padded seq_lens/cu与全局RNG。验证步骤包括同步/额外前向，不能当吞吐结果。通过后再独立运行不含数值观测的4K/1K服务对照及新Math。

只追加新folder，不改91份已测诊断来源；当前正式best4093.81及历史Math102/105保持。用户原5处WIP保留，无新16K/push/agents/MetaX改动，GPU硬停2026-10-19 23:58+08。
