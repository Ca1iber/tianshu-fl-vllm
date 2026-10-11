# 天数最佳配置短 profile（准备中）

旧 profile 属于优化前的原生 Attention，不能直接作为新 Hybrid 配置的瓶颈结论。这里保留独立脚本，使用已经通过质量门槛的策略，记录实际源路径/SHA、批次预算、Graph 模式与 GPU 空闲状态。

在短预算实验完成后，串行测量4096输入/128输出的16/64路请求，用torch profiler记录GPU kernel时间分布。profile产生的开销不能当作官方吞吐成绩；Graph关闭的诊断也不能代替正常Graph运行结果。启动前后只检查和清理脚本自身服务。

已核实旧trace覆盖存在缺口：Graph开启的baseline C16只记录1386次Attention（全为2D），Graph关闭则记录6720次并含2072.44ms的3D Decode；两者2D时间约14136.6ms相同。因此正常Graph trace的百分比仅针对已记录内核，不能称为完整GPU时间占比。新分析显式记录覆盖未核实，并保留Graph关闭的独立诊断模式；两轮相同工作负载预热后再采样，记录调优日志。
