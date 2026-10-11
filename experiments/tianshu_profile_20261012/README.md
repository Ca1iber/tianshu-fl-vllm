# 天数最佳配置短 profile（准备中）

旧 profile 属于优化前的原生 Attention，不能直接作为新 Hybrid 配置的瓶颈结论。这里保留独立脚本，使用已经通过质量门槛的策略，记录实际源路径/SHA、批次预算、Graph 模式与 GPU 空闲状态。

在短预算实验完成后，串行测量4096输入/128输出的16/64路请求，用torch profiler记录GPU kernel时间分布。profile产生的开销不能当作官方吞吐成绩；Graph关闭的诊断也不能代替正常Graph运行结果。启动前后只检查和清理脚本自身服务。
