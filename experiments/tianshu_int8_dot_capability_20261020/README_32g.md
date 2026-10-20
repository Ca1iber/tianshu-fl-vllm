# INT8 dot能力检查：仅准备，非性能试验

后续可能探索每token/head K量化配合逐query/head Q量化，直接用整数QK乘法消除读K后的浮点解码。先用真正天数Triton后端确认16×128与128×16/32的INT8 dot是否可编译、实际INT32乘积是否逐位正确。Q按行nearest-even量化；两个同Graph相位变化Q/K/尺度，独立CPU整数积和FP32缩放参考，RNG检查。

最坏128×128×128=2097152，INT32累加不溢出。本探针不含分页、softmax、V读写、完整attention、模型或Math，也不计时，因此即使通过也只证明这个入口，不证明端到端收益、格式适用性或硬件INT8峰值。编译/初始launch错误保留，不把一次失败直接归因为硬件不支持。必须等当前服务测试释放GPU后运行。

实际v2已通过两种尺寸、两相位：Q量化、Q尺度、INT32乘积均与独立CPU逐位一致，浮点缩放在容差内，RNG不变。初次仅因错误引用CUDA命名空间导入失败，旧代码/日志留原始目录；核对安装源码后改为Corex的libdevice，没有删除失败原件。没有计时，因此仍不宣称性能收益。

真实驱动查询记录：BI-V150 32GiB、16个multiprocessor、warpSize64、max_num_regs262144、max_shared_mem131072，频率等原始值见GPU_GATE_32g.json。major/minor是此后端上报值，不按NVIDIA架构名称解释。下一阶段若探索每token/head K尺度和完整INT8 QK attention，需要全新的格式、数值/Graph、服务和Math验证；当前正式4K5101.45/16K3213.11保持。

单warp独立入口也已通过两种尺寸、两相位，逐位Q量化/尺度/整数积与CPU一致。16/32列分别101/122寄存器、0 spills，编译元数据num_warps=1；见GPU_WARP1_GATE_32g.json。此处仍没有计时，也未覆盖完整attention。
