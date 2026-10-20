# INT8 GateUp + 原生SiLU标量融合组件

此前BF16 affine融合全部负收益。本次使用已验证W8A8、32/64整tile、两个INT32投影累加器，投影严格恢复原BF16舍入，再调用FlagGems真实PointwiseDynamicFunction._scalar_fn，不另写sigmoid近似。

真实layer0 W原始BF16/packed/scales完整签名与合格模型一致。五组实际M/stride、24配置，每配置三次变A/W数据同Graph。对照采用当前最佳原整数GUP配置加真实native SiLU，双方计时都包含真实补齐量化，权重打包均排除。原实际计时输出要求完整BF16逐位相同，诊断专门化额外验证全INT32与BF16投影；两类编译寄存器和hash分开记录。CPU独立重算A/W量化、INT64点积、投影和激活样本。

仅在正确且组件有收益后才考虑真实LlamaMLP调用链集成；不修改现有5482.79配置或正在跑的padding服务。完整raw tensor/日志仅容器，GitHub仅源码/报告/关键计时与证明。
