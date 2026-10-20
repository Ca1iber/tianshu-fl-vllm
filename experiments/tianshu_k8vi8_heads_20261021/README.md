# K8 + V逐token/head INT8独立组件

使用当前合格f1df成对K读取器，精确扩展两处mode判断，使K软件8编码及成对读取同时支持mode0和mode2。V为INT8，复用vLLM原生mode2的P逐token/head缩放；K scale实际存储1，不改变K编码。

原生FullAttentionSpec.page_size_padded=8448，K/V各4096字节，K/V FP32 scale各128字节；全部typed view共享真实raw分配，stride含全部元数据，较K8V16的12288字节减少31.25%。没有额外未计入容量的scale张量或隐藏BF16缓存。

三种V格式使用同一K代码、同一页表、同一固定launch策略。10个batch/context组件场景，3次真实数据/长度/scale变化共用Graph；时延包括实际writer+量化+scale写入+reader+reducer。逐物理字节检查包含所有padding/未用页/负slots，CPU独立复核实际writer整数、scale和保存的attention输出。40/16K是单层组件点，不是完整16K benchmark。

这是未接入ModelRunner、无prefill生命周期、无Math和服务吞吐结论的组件实验。CPU验证后排队在本轮正式4K及GEMM补齐组件之后，避免争用GPU。只有组件获益才考虑全流程接入，并必须重新验证准确率。
