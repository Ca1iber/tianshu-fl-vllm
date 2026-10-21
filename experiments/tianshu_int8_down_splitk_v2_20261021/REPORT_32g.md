# Down INT8确定性SplitK完成：数值通过，4K性能负收益

2026-10-21 BI-V150 32G。v1保留首次tail编译失败；v2只将分段CHUNK在实际CPU helper中计算后传入constexpr，算法、9配置和数值门不变。真实layer0 Down [2048,6144]，M129/1024/2048/4096，共36组合全部编译成功。GPU三阶段完整INT32、BF16/原unsplitINT8逐位、partial求和、边界与RNG断言通过；CPU独立重算108组选中INT64结果、每段选中partial、全部134498304个A量化值，以及54份小尾部完整输出/partial。CPU没有重算每个大矩阵的完整输出；完整大输出对照在冻结GPU探针中执行。

7轮四路旋转，每轮20次热图重放；包含A动态量化、partial完整读写及归并，静态W打包排除。BF16 mm保留现有M2048特殊实现；同次另测aux BF16 linear及原始M4096最佳tile的unsplit INT8。下表候选为每个M的9选1探索最快值，不能视为独立holdout。

| M | 原BF16 mm ms | 原unsplit INT8 ms | 新partial+归并 ms | 新选中splits |
|---:|---:|---:|---:|---:|
|129|.105716|.423849|.197827|2|
|1024|.557180|.285468|.311900|1|
|2048|.655920|.537435|.597027|1|
|4096|1.103415|1.039579|1.157647|1|

在主要M4096上，最快新实现仍为splits1（已经包含中间INT32写入和归并），比同次BF16耗时高4.91%，比原unsplit INT8高11.36%。同一128×128 tile下splits2/3/4/6分别1.182458/1.280045/1.373027/1.571452ms，producer均168寄存器、0 spill；128×512 tile有81 spill。未采集单独producer/reducer时间，不能把全部差异归因某一阶段。

这组配置未带来当前4K所需的组件改善，不接入模型、不做新Math或正式benchmark。不存在新的模型成绩/16K成绩；原推荐仍5586.28 total tok/s、TTFT3780.63ms、Math102/105。最大单算子partial workspace201326592字节，尚未做任何模型显存安全或精度认证。

原始PT/日志/快照仅容器/workspace/benchmark-results/20261021_int8_down_splitk_v2_GPU_tianshu；Git只存代码、本报告和关键JSON/CSV。CPU核验日志在同前缀CPU_tianshu/verify.log。
