# 同一K4格式和数值：两个字节一次load，解码四个D

保留byte-load位构造版及其全部负收益。它消除了K的4KiB额外共享内存与sitofp，但C64仍慢约38%。新研究问题是能否通过一次uint16读取四个nibble，减少byte-load与展开布局成本。仅改K load和对应四值展开；量化规则、writer、页面、K scale后QK、V、softmax/reducer和grid/launch配置一致。

先实际生成helper穷举全部65536个word，每个对应4个BF16值，包括code0=-8；连续/跨步、两阶段同Graph共1048576输出逐bit检查，保护input/guard/RNG。再完成同样8场景、9次交替20回放的完整writer+reader组件；全输出GPU FP32及独立CPU FP64、全部原始输入哈希与raw缓存字节重建，记录代码生成、二进制与spill，spill不作为数值正确性条件。

预先定义本研究的模型接入候选条件：C64 4096、5120、ragged/strided三个官方并发代表场景，完整组件均至少提升1%；其余C8/C33/长上下文/capture转换为必要数值与性能诊断，完整记录不隐藏回退。该门槛只用于本次word研究，旧byte/sitofp变体的失败不改写。若通过，再做真实引擎页面接入、prefill解码及全Math>95与正式4K；组件或容量公式不能替代模型结果。当前r5继续保留。
