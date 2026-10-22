# W4A16/W8A16 第一轮实际组件结果

数值组件检查和独立 CPU 审核通过，未跑模型/MATH/正式 benchmark，未采用。真实第0层 Down 与 LM head、合成 BF16 activation，含量化权重读取/解包；不包含一次性打包。7组轮换、每组20次热 Graph，取时间中位数，再从3组预定 tile 中取最快。BF16 为同进程 FlagGems mm_out；W8A8 为已有动态 A8 + NT 整数路径。

|角色/M|格式|最快 tile|BF16 ms|W8A8 ms|候选 ms|候选/BF16 耗时|
|---|---|---:|---:|---:|---:|---:|
|down/8|W4_sym128|0|0.092837|0.136017|0.951102|10.24x|
|down/8|W4_aff128|0|0.092535|0.136027|1.213510|13.11x|
|down/8|W8_channel|0|0.092513|0.135806|0.144205|1.56x|
|down/64|W4_sym128|2|0.059234|0.069421|1.627814|27.48x|
|down/64|W4_aff128|2|0.059302|0.069066|1.688009|28.46x|
|down/64|W8_channel|2|0.058702|0.066833|0.350848|5.98x|
|lm_head/8|W4_sym128|0|0.817489|0.866846|14.739064|18.03x|
|lm_head/8|W4_aff128|0|0.817196|0.865903|15.154106|18.54x|
|lm_head/8|W8_channel|0|0.818339|0.870009|1.885439|2.30x|
|lm_head/64|W4_sym128|2|1.038291|0.619801|21.615080|20.82x|
|lm_head/64|W4_aff128|2|1.024775|0.614394|23.413246|22.85x|
|lm_head/64|W8_channel|2|1.032874|0.618744|4.314962|4.18x|

全部12组角色/格式/M组合均慢于 BF16。不能由此推断所有 INT4 实现或天数硬件必然无收益；这是该 NN 打包、逐元素解包到 BF16、固定 tile 的组件结果。下一独立候选改成 packed 字节只读一次、固定移位后交织。另一独立候选将每通道 W8 scale 移至完整 dot 之后，需新的数值参考。

独立 CPU 从只读原始 checkpoint 重新量化，27个全量 decode 相位 SHA一致，135个 artifact 通过；87777个 selected FP64输出、257850081个全部输出 BF16 cast、所有 activation量化及 selected W8A8 INT32/BF16准确。原始 checkpoint 和冻结源均未变。V1 的 signed-zero 参考错误及独立诊断已另存，V2 未放宽非零误差。

payload节省不能当成引擎显存节省：当前仍需 BF16 fallback，若接入新权重副本，必须重做 profiler memory accounting。没有新模型正确率或正式吞吐成绩，当前发布仍5695.95 total tok/s。

原始目录 `/workspace/benchmark-results/20261022_quant_weight_component_v2_GPU_tianshu`，仅容器保留完整数组。代码、报告和关键CSV/JSON推天数 GitHub。
