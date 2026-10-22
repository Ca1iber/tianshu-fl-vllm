# W4A8 NN 布局与分阶段计时：矩阵本体仍慢

沿用NT版的A8和group128 INT8 dot/FP32缩放，仅把W4 packed权重换为 `[K,ceil(N/2)]`，成对读取相邻N后交织为BK×BN右操作数。12tile、2格式、真实Down/M64与strided/tail共48配置、144相位，数值独立审核全部通过：12次完整decode、178776个FP64 selected结果、9597528个完整cast、1566672个实际INT32 partial精确验证。只含GEMM的图每相位输出与完整图逐位相同，只含A8的图q/scale与原实现逐位相同。

|Down/M64 最快tile8|完整路径 ms|量化单独 ms|矩阵单独 ms|同进程BF16 ms|
|---|---:|---:|---:|---:|
|对称W4 group128|0.147719|0.004624|0.143239|0.059135|
|仿射W4 group128|0.151180|0.004643|0.146802|0.058882|

tile8=BM64/BN64/BK128、8warps/1stage。完整路径保留A8量化成本，7轮轮换×20热Graph；量化图32个节点除32，其余单节点。分开捕获/测量会有噪声，三项中位数不强制精确相加，差值不视为负开销。

这版NN矩阵本体仍远慢于BF16，即使完全省掉独立量化的约4.6us，也无法弥补差距。不能把这里的矩阵单独计时当作模型速度；完整路径比NT版更慢，也不能推断其他NN实现一定如此。不接模型，不跑新Math/正式benchmark，不采用。原checkpoint、冻结源码、输入、guards/RNG均保持。

完整数组仅容器 `/workspace/benchmark-results/20261022_w4a8_group_nn_stages_GPU_tianshu`，代码/报告/关键CSV与审核JSON入GitHub。当前正式发布仍5695.95 total tok/s。
