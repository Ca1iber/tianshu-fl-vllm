# W4A16 NN每组点积后缩放：组件仍落后

30配置、90相位artifact、18个完整format decode相位与独立CPU审核通过；原checkpoint和冻结源未变。真实Down/LMhead M8/M64，BF16合成activation，不是模型benchmark。

每K128组先dot(A_BF16, centered_integer_BF16)，再乘FP32 scale累加；完整GPU参考用未做BF16 round的q*scale矩阵，独立CPU从原weight重算FP64 selected点积，不借用上一版BF16 rounded矩阵参考。每个scale/解包/dot都在计时Graph里，一次CPU打包和诊断不计时。

|角色/M|格式|tile|BF16 ms|W8A8 ms|候选 ms|候选/BF16|
|---|---|---:|---:|---:|---:|---:|
|down/8|W4_sym128|0|0.092983|0.136008|0.223681|2.41x|
|down/8|W4_aff128|0|0.092275|0.135910|0.219272|2.38x|
|down/64|W4_sym128|2|0.058887|0.069172|0.462430|7.85x|
|down/64|W4_aff128|2|0.058817|0.069538|0.464727|7.90x|
|lm_head/8|W4_sym128|0|0.818863|0.867479|2.944890|3.60x|
|lm_head/8|W4_aff128|0|0.817998|0.864304|2.923444|3.57x|
|lm_head/64|W4_sym128|2|1.041974|0.622776|5.905989|5.67x|
|lm_head/64|W4_aff128|2|1.037285|0.624845|5.673891|5.47x|

三个预定tile各7组轮换、每组20次热Graph，中位数选最快；全部真实形状/格式仍慢于同进程BF16，M64 LM还慢于现用W8A8。未接模型、未跑新Math或正式benchmark，未采用。后缩放没有解决此NN实现的全部成本，不能把慢的原因仅归于weight scale乘法；还需NT布局/实际计数器验证。

raw `/workspace/benchmark-results/20261022_w4a16_group_postscale_GPU_tianshu`，大数组仅容器，GitHub留代码/报告/关键结果。现行正式发布保持5695.95 total tok/s。
