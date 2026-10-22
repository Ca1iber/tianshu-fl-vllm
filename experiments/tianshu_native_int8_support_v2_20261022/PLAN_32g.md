# CoreX 原生INT8 GEMM支持性 V2：M64

V1先在M8被host要求M>16拒绝，没有执行GPU整数dot；源码/失败保持。V2只检查实际主要decode M64，K2048/K6144、N256，NN contiguous/NT-transposed，3相位改变A和W。每个完整INT32输出由CPU INT64精确重算，Graph out捕获、guards/输入/RNG均检查。

不读模型、不计时、不宣称模型质量或速度，不为M8静默padding。实际接口存在不等于实际GPU支持；失败完整保留。源码/计划先冻结，前置GPU空闲，单任务120秒，今日23:58停止，仅容器运行，关键结果推天数GitHub。后续若支持仍须真实GateUp bank的A8+原生INT32+缩放/SiLU完整成本对照。
