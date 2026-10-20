# KV后续方向：先排除已测路径

只读审查，当前新prefill Math/正式4K的407运行来源保持不变。旧软件E4M3 K/V格式已有全新Math102/105通过（soft8_engine_v2_20261019/MATH_REPORT_32g.md），不能把后续选择BF16 V的原因误写为准确率不合格。当前K8/V16主要是已测性能更好。

K8VI8Heads V2（tianshu_k8vi8_heads_v2_20261021/source.py）已经启用native scaled-P：先把每token/head的V_scale乘到P，再dot(P_v,V)，并没有对整块V做多余逐元素scale。故“把V_scale移到P侧”不能再次当新优化跑一遍。当前同配置C64/4096实测K8V16 0.539952ms、INT8V 0.751586ms，+39.19%耗时；8448-byte页比12288-byte页少31.25%，节省存储不等于更快。

原始 /workspace/benchmark-results/20261021_k8vi8_heads_v2_GPU_tianshu/result.json 的producer记录：C64/4096、IS_3D、segments4、T16、w1相同，K8V16 176regs、INT8V 189regs、software8 V238regs，均0spill。该结果只说明资源不同，没有硬件stall/counter，不能单凭register数解释全部性能差距。三相位、完整writer/scale/页保护和独立CPU参考已通过，重复不变参数没有价值。

可能后续研究成对/向量化V读取、移除重复K_scale=1的运行加载、或更大原生页的实际reader；这些均未实现、未跑GPU、未证明收益。需要保留原cache布局/分页/尾块/Graph变数据检查，先组件后模型，再新Math；不能从较小索引或更少字节直接宣布收益。当前优先完成已运行的新prefill质量/正式4K，再做新准备的真实Down权重组件。
