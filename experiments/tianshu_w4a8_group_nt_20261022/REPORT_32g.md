# W4A8 NT 分组整数 dot：数值通过，速度未超过 BF16

本版采用实际第0层 Down 权重，对称/仿射 W4 group128、NT packed `[N,ceil(K/2)]`，解包 INT8 后执行整数 dot，每K组用 FP32 weight scale 累加，最后乘 A8 row scale。完整路径包含原 A8 量化；不是原生 INT4 指令，也不是完整模型评测。

12组预定tile×2格式×真实Down/M64和strided/tail，共48配置、144个变化输入相位均通过。独立CPU从原checkpoint重建格式，完整decode12相位、178776个FP64 selected输出、9597528个BF16 cast、1566672个实际 INT32 partial 与 INT64 dot 逐项一致；A8、W8A8整数控制、输入/guards/RNG和冻结源均保持。原checkpoint SHA前后相同。

|Down/M64|最快完整路径 ms|同进程 BF16 ms|同进程 W8A8 ms|
|---|---:|---:|---:|
|对称W4 group128|0.075430|0.058882|0.069203|
|仿射W4 group128|0.097982|0.060359|0.069337|

两者最快tile均为BM64/BN64/BK128、8warps/1stage。7轮轮换、每轮20次热Graph，中位数；CPU一次性打包与诊断不计时，运行时A8量化计时。对称格式完整耗时比当前BF16多约28.1%，不接模型、不跑新Math/正式benchmark、不采用。此结论只覆盖本次shape和实现。

下一版单独改变为NN packed存储并拆分量化/GEMM计时，见 `../tianshu_w4a8_group_nn_stages_20261022/REPORT_32g.md`。完整数组仅容器 `/workspace/benchmark-results/20261022_w4a8_group_nt_GPU_tianshu`；关键JSON/CSV及源码入GitHub。
