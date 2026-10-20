# INT8 prefill grouped-M：完整组件复核通过

2026-10-21，BI-V150。8种M×6分块×group1/group8，共96配置、288次变A/权重符号/scale，GPU完整INT32/量化BF16、CPU全部A和原checkpoint选中W重新量化、实际GPU选中整数INT64、guard、输入/RNG/变数据Graph检查通过；没有编译失败。CPU tile几何另外480种覆盖无重复无越界。每配置七轮交替、每轮20 Graph，含完整A动态量化；静态W预转置排除。

| M | BF16 linear ms | 最佳INT8 ms | group | BF16/INT8-1 |
|---|---:|---:|---:|---:|
| 128 | 0.131707 | 0.095588 | 1 | +37.79% |
| 129 | 0.192439 | 0.371052 | 8 | -48.14% |
| 256 | 0.214652 | 0.144458 | 8 | +48.59% |
| 512 | 0.318140 | 0.253303 | 8 | +25.60% |
| 1024 | 0.582697 | 0.502778 | 8 | +15.90% |
| 1152 | 0.768856 | 0.593284 | 8 | +29.59% |
| 2048 | 1.171426 | 1.003310 | 8 | +16.76% |
| 4096 | 2.364668 | 2.012801 | 8 | +17.48% |

这是12选一的探索性组件最佳。17.48%是换算组件吞吐提升，对应耗时约降低14.88%；不是模型总吞吐提升。group1/group8同tile成对值另见GROUP_COMPARISON_32g.json，不把跨轮不同BF16时间或不同tile相减当成单独group8效果；没有采集硬件cache计数器，不能把缓存复用动机当成已证实因果。

M129全部INT8候选慢于当前BF16，最优还出现21 spill；仅凭这一项不能断定全部原因。模型候选将按128对齐的M用于INT8，其他M保持BF16。对齐中间M用最近上方实测tile，启动前逐个数值预热，不冒称所有中间M已经实测加速。

此组件仍非模型/Math/正式benchmark。新模型配对两组使用相同纯decode M8/M64 NN布局，只改变大M prefill及混合批次；混合批次中全部行都会走同一矩阵投影，不能声称其中decode行单独保持BF16。最终需要4K小测、全新105题Math严格>95%，再正式4K。无常规16K。

GPU_GATE_32g.json、GPU_ROUNDS_32g.json、GPU_COMPARISON_32g.json、KEY_TIMINGS_32g.csv及GROUP_COMPARISON_32g.json保留关键结果；原始PT、源码快照和完整result仅容器 /workspace/benchmark-results/20261021_int8_gateup_prefill_grouped_GPU_tianshu。
