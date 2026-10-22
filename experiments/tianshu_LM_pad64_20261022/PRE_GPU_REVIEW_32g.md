# 首次GPU前记录设备二进制

为后续可能的服务集成增加每个M点实际quantizer/GEMM的CUBIN SHA、cache key、寄存器/spill/shared元数据；固定GEMM64与运行时M量化的数值/性能计划不变。本组件此前未运行GPU。纯CPU准备新runner_v2，执行清单与源重新冻结，不修改上一已完成LM转换实验。
