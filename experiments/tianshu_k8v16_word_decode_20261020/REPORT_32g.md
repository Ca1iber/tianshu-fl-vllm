# K word先解码和pair warp2：全数值通过，明显负收益

2026-10-20，BI-V150 32G。对照是已组件通过、4K N64服务提升2.5671%的两字节pair_rows。候选四字节读取先恢复四份BF16再组K矩阵，分别行/列方向；另一路只将pair producer改为warp2。缓存位模式、writer、V BF16、分段、tile16和reducer保持。

12场景×4组×两相位共96份完整输出FP32复核、256独立CPU向量、5947392个输出元素复核通过；改变数据重放同Graph、整个缓存/边界/RNG亦通过，实际三候选全部逐位等于pair控制。warp2逐位一致是本次实测，预先只把它作为诊断而没有以此替代准确率。432组交替设备event，每组20重放。

|场景|pair_rows控制 ms|word_rows ms|word_columns ms|pair warp2 ms|
|---|---:|---:|---:|---:|
|B64/4K|0.55252266|2.58345623|1.01840248|1.32433596|
|B64/5K|0.68259540|3.22983856|1.26705961|1.64609241|

B64/4K三候选倒数时间比分别-78.6130%、-45.7461%、-58.2793%；寄存器176→130/115/132，spill均0。寄存器更少没有带来速度收益，无法仅据此归因具体硬件瓶颈。四字节组装并不优于两字节，warp2也不采用，不追加这些候选的服务/Math/正式benchmark。

最终候选仍为pair_rows/warp1，转入新Math和仅完整4K。GPU_GATE_32g.json与GPU_ROUNDS_32g.csv为关键结果，完整张量、编译IR/PTX、CPU参考与原始事件留/workspace/benchmark-results/20261020_k8v16_word_decode_GPU_tianshu，不上传GitHub。
