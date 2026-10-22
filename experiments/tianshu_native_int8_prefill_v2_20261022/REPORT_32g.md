# 原生 INT8 完整 GateUp：大M出现组件阳性，尚未接模型

在原资格三个完整cubin逐点严格匹配后，真实第0层W8 GateUp、M128/512/4096，2布局×2epilogue行分组，3变化输入相位均完成。GPU全INT32、BF16投影与实际计时激活输出与当前实现逐位相同；独立CPU重建checkpoint量化签名、全A8、4608个selected INT64 dot、174587904个完整投影BF16元素hash，以及selected FP64激活参考均通过。36候选相位比较、9共同artifact，guards/输入/权重/RNG、冻结源/缓存、原模型SHA不变。

|M|当前完整路径ms|NN/ROWS1 ms|NN/ROWS4 ms|NT/ROWS1 ms|NT/ROWS4 ms|
|---:|---:|---:|---:|---:|---:|
|128|0.089555|0.093215|0.091825|0.098285|0.096431|
|512|0.261038|0.232177|0.232575|0.241246|0.241415|
|4096|1.957708|1.493048|1.512799|1.462070|1.485458|

7轮轮换×20热Graph，所有候选含A8量化、原生INT32写读、缩放/BF16投影和原SiLU，排除CPU静态打包及诊断。M128仍负。M512 NN/ROWS1逆耗时+12.43%；M4096 NN/ROWS1+31.12%，NT/ROWS1+33.90%。只是一层合成activation下的组件；未测完整模型性能或新Math，不能升级正式成绩。

优先继续NN/ROWS1的M512/M4096，复用现有NN W8 bank而无需保留另一物理W布局。NT微测稍快，但若同时保留fallback NN bank会增加约1GB全42层权重payload，尚未验证引擎内存或KV容量，不能只选微测最快而忽略成本。最大单INT32 workspace201326592字节也必须被真实profile_run内存预算计入，不能在KV cache定额后偷偷新增不计费常驻缓冲。

下一步按已固定的NN/ROWS1先做42层真实bank冷权重、GateUp+Down全MLP holdout；若仍阳性，再独立模型profile/fake/FX/Graph覆盖、内存审核、小4K同期对照和fresh C8/C64 Math，达到晋升要求才正式N256。M8、M64与其他M保持现用实现。

V1完整binary拒绝和CPU首次只读旧cache准备失败均保留。后续按两个设备AST相同源码路径复现3个正式binary，M128/V1、M512/M4096/V2，全cubin匹配门槛未放宽；M128 ELF所有ALLOC section一致解释debug路径差异。cache/group实际来源和复制索引均hash冻结，未改旧资格cache。

raw `/workspace/benchmark-results/20261022_native_int8_prefill_v2_GPU_tianshu` 只保留每相位一份共同INT32和完整hash/选点，避免重复相同大数组。代码、报告、关键JSON/CSV推天数GitHub；完整数组/缓存/日志仅容器。正式发布仍5695.95 total tok/s，现有模型和沐曦仓库未改。
