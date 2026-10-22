# W8A16 NN点积后统一缩放：仍负优化

15配置、45相位artifact、9个完整decode相位及独立CPU审核通过。只把通道scale移到dot后，仍使用NN权重布局；这是新的数值顺序，不借旧Math。checkpoint与冻结源不变。合成BF16 activation +真实第0层Down/LMhead权重，非模型benchmark。

|角色/M|tile|BF16 ms|W8A8 ms|本候选 ms|候选/BF16|
|---|---:|---:|---:|---:|---:|
|down/8|0|0.092791|0.136025|0.144354|1.56x|
|down/64|2|0.059298|0.068205|0.338771|5.71x|
|lm_head/8|0|0.819617|0.866193|1.877103|2.29x|
|lm_head/64|2|1.037500|0.621847|4.094505|3.95x|

7组轮换、每组20次Graph replay，包含读取/整数码转BF16/dot/scale，排除一次性CPU打包和诊断store。全部实际形状仍落后BF16，M64 LM还落后现用W8A8；不接模型，不跑MATH或完整benchmark。不能凭本结果推断其它权重布局均无收益；NT布局是另一个独立候选。

CPU参考从原checkpoint重新量化；完整decode比对仍是格式检查，实际GEMM用未做BF16 round的q*scale理想参考，独立FP64点积/全部输出cast和原W8A8整数结果审核通过。完整计数见INDEPENDENT_CPU_GATE_32g.json，raw `/workspace/benchmark-results/20261022_w8a16_postscale_GPU_tianshu`，大数组仅容器。当前正式发布保持5695.95 total tok/s。
