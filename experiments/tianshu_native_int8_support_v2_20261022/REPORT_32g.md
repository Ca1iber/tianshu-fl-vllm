# M64 原生 INT8 GEMM 支持性通过

已安装CoreX torch2.10.0的torch._int_mm(out)在BI-V150实测可用。M64/K2048和K6144、N256，NN contiguous和NT-transposed两种权重布局，共4案例×3相位。196608个实际INT32输出全部与CPU INT64点积一致；同一Graph在A/W变化后重放、guard、输入/RNG不变均通过。CPU比较在GPU应用进程内完成，不称为独立进程审核。

没有计时，没有读取checkpoint，也没有模型质量或性能成绩。接口M8受M>16限制，V1失败保留；后续只对实际M64 GateUp做真实权重全成本组件，不能由本支持性测试声称原生INT8已经更快。raw在容器 `/workspace/benchmark-results/20261022_native_int8_support_v2_GPU_tianshu`，源码/计划/关键gate保留GitHub。
