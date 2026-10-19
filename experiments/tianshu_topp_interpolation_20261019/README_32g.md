# Top-P 探测位置：仅 CPU 可行性检查

控制组服务运行时，在本地 CPU 检查另一项候选：用带括区间保护的 log-domain 插值/Illinois 更新选择概率阈值探测点，保留原精确边界终止条件。还未生成、编译或测试 GPU 内核，也没有采样、RNG、Math 或服务结果。

16 个 FP64 CPU 分布（8192/130560、normal/peaked/uniform/duplicate、每种2种子）中，几何探测平均12.75轮，secant9.9375，Illinois9.8125，周期几何保护11轮；最终边界前缀计数均匹配平衡FP64前缀参考。CPU运算顺序、outlier buffer和FP32归约均未复现，因此不能把这些数量当实际GPU扫描次数或性能收益。

第一版 check_search.py 的大uniform参考使用逐项FP64加法，累计误差使参考计数多1，四种算法在这两个用例均显示不匹配。原源与原始输出保留于service-results/20261019_tianshu/topp_interpolation_CPU.json。独立check_search_v2.py只将参考改为math.fsum前缀二分；搜索算法未改，16/16匹配，结果CPU_SEARCH_FEASIBILITY_32g.json保留执行源码SHA。该修正不代替生产参考或放宽GPU门槛。

若后续实施：独立生成源码，TopK AST、Top-P概率值、精确重复值处理、32轮上限与1e-12区间门槛保持；必须经真实FP32与CPU 1e-6参考、变输入Graph与真实API/RNG检查，之后才做服务配对。目前未采用。
