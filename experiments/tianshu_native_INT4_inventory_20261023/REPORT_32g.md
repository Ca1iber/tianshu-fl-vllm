# 原生通用INT4 GPU四例功能通过

冻结f8d06ba，按顺序等待prefill短测结束后在空闲BI-V150上首次执行。M64 K256 N256，q=7/8/9/10，BF16 A=1/8、scale=1/8、zero=0、group128、innerK8。GPU输出分别-4/0/4/8，独立CPU逐格核验65536个BF16值全部精确一致；原先预声明参考不变。输出SHA/实际错误状态和源码SHA保留，未打开checkpoint，未测试真实权重、模型质量或性能。

实际PyTorch后端发出提示：通用_weight_int4pack_mm仅做功能兼容，推荐_weight_int4pack_mm_dotsaddz获得更好性能。该提示指引后续测试方向，不能据此宣称dotsaddz在本卡已经更快。另行的非恒定矩阵实验已确认通用高nibble偶K布局及AWQ permuted+packed-zero布局正确；其余假设的真实不一致全部保留。

原始数据`/workspace/benchmark-results/20261023_native_INT4_smoke_GPU_tianshu`仅容器；FUNCTIONAL_GATE保存独立结果。本实验不改变正式r3版本或性能成绩。
