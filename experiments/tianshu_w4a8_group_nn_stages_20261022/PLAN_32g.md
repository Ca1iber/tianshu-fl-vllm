# W4A8 NN存储和量化/GEMM分阶段测量

NT版48配置完整GPU/独立CPU通过，1566672个实际selected INT32 partial完全相同，最快Down/M64对称W4为0.075430ms，BF16为0.058882ms，仍负优化。激活A8单独微测M64/K6144旧量化器约4.58us；不能仅靠省去量化解释全部差距。

本版只把packed权重改为[K,ceil(N/2)]，相邻N交织后直接形成BK×BN INT8右操作数，其余A8 RNE、group128、整数dot/FP32缩放、12tile和输入保持。新格式需完整decode/selected integer partial重新审核。目标真实第0层Down/M64，另strided/tail；不改变模型或原weight。

每图3次输入/码/scale变化、全量GPU浮点参考/输出cast，独立CPU从checkpoint重新量化并逐块INT64点积。另捕获只含GEMM的图与32次量化的图；每相位只含GEMM的结果需与完整图逐位相同。7轮轮换×20 replay，quant_only除以32；完整图仍包括量化成本，不能用matrix_only替代模型收益。检查矩阵本体是否仍慢于BF16，再决定有无融合价值。

来源、raw和失败保留；源冻结再运行，actual GPU idle，单任务30分钟、本日23:58截止，每日06:00–24:00授权。代码报告关键结果GitHub，原始数组仅容器。阳性组件>10%才进一步考虑模型显存/新Math/正式benchmark。
