# 验证控制器CPU检查

CPU检查通过，当前screen运行来源407份完整SHA不变；原105题/正式4K工作负载控制器仅改profile名称，其他字节保留。旧dense质量审计仅两处BF16 prefill覆盖语义适配，其他文本完整保留；baseline quality audit和batch policy字节相同。新增审计要求42层实际INT8 prefill、NN物理bank与原SHA、M8/M64实际捕获、相同warm/RNG/线程和SQL真实执行，Math按C8、正式按C64，不伪造另一批次请求覆盖。

控制器与运行来源分别冻结。此证明没有运行新的Math/正式benchmark，不能当准确率达标。screen若无足够收益或仍现场autotune，不启动这套长评测。
