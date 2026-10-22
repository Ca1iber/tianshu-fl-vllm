# 原生INT8 GateUp M64：全位相同，完整路径仍稍慢

真实第0层Gate/Up BF16 checkpoint只读并复核原W8 bank签名。当前正式M64 two_dot融合BM64/BN64/BK128/W4/S1的实际binary严格匹配控制；候选为相同A8+torch._int_mm(out)+独立投影缩放/原FlagGems SiLU。2种A stride×2种权重布局×6种epilogue=24个候选计时点，3变化相位共72份artifact。

GPU完整INT32、BF16投影与实际计时激活输出均逐位相同当前路径；独立CPU从checkpoint重新量化、全A8、36864个selected INT64 dot、56623104个完整投影BF16元素以及selected FP64 SiLU参考均通过。guards、A/W/scale、RNG、冻结源不变；原checkpoint前后SHA一致。数值正确不宣称新模型质量。

|M64 stride/布局|当前完整路径ms|候选最优完整路径ms|最优epilogue|
|---|---:|---:|---|
|连续/NN|0.063122|0.066165|BLOCK1024/W8|
|连续/NT|0.063122|0.069962|BLOCK1024/W4|
|stride2055/NN|0.062905|0.066316|BLOCK512/W8|
|stride2055/NT|0.062905|0.070248|BLOCK1024/W4|

7轮轮换×20热Graph，计入量化、原生INT32写读与独立epilogue；静态CPU打包和诊断排除。选优仍慢，不接模型、不跑新Math/正式benchmark，不采用。M8原生接口要求M>16，保持原实现，没有隐藏padding。不能据M64负结果断言较大prefill M也负收益，下一独立版本见 ../tianshu_native_int8_prefill_20261022。

每个原生M64 INT32 workspace3145728字节，NN/NT各W8副本25165824字节。组件包含两布局只为比较，尚未证明模型中NT布局替换/保留会造成多少净显存，不能跳过native profiler直接部署。完整数组仅容器 `/workspace/benchmark-results/20261022_native_int8_gateup_GPU_tianshu`，关键JSON/CSV、源码报告推天数GitHub，现用版本不变。
