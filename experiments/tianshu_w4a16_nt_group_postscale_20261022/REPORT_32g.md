# W4A16 NT分组后缩放：数值通过，仍负优化

30配置、90相位artifact、18个完整格式decode相位与独立CPU审核通过，checkpoint/冻结源码保持。真实Down/LMhead M8/M64，原始BF16 activation使用合成样本。新的NT布局和2/4warp tile整体筛选，不将综合变化归因单个因素。

|角色/M|格式|tile|BF16 ms|W8A8 ms|候选 ms|候选/BF16|
|---|---|---:|---:|---:|---:|---:|
|down/8|W4_sym128|2|0.092346|0.136025|0.113563|1.23x|
|down/8|W4_aff128|2|0.092785|0.135951|0.125507|1.35x|
|down/64|W4_sym128|2|0.059154|0.069204|0.209211|3.54x|
|down/64|W4_aff128|2|0.058847|0.069047|0.220861|3.75x|
|lm_head/8|W4_sym128|0|0.818586|0.867186|0.898102|1.10x|
|lm_head/8|W4_aff128|0|0.817744|0.868915|1.040310|1.27x|
|lm_head/64|W4_sym128|2|1.039650|0.621074|2.371455|2.28x|
|lm_head/64|W4_aff128|2|1.032932|0.616643|2.640391|2.56x|

每个预定tile 7组轮换、每组20次热Graph，中位数选最快。计时包括实际加载/解包/BF16 dot及分组FP32 scale，不含一次CPU打包或诊断。全部真实形状/格式仍慢于BF16，M64 LM还慢于现行W8A8。当前不接模型/Math/正式benchmark，不采用。

数学参考是未做BF16 round的分组q*scale，在完整GPU FP32参考及独立CPU FP64 selected输出上审核，不借用旧BF16重建矩阵。格式decode另行全量位级检验；全部output cast、A8量化、selected W8A8整数结果、guards/RNG/输入不变通过。raw `/workspace/benchmark-results/20261022_w4a16_nt_group_postscale_GPU_tianshu`，大数组仅容器。

下一步实际分析器计数器定位；原第一个IXKN采集应用正常、外层退出139且导出文件两次import失败，暂不引用其中计数器，另外目录尝试直接CSV。保持正式发布5695.95 total tok/s。
