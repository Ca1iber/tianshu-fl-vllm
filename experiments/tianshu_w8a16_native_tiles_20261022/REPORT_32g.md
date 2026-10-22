# 对齐原生线程/流水后：W8A16明显变快，仍不采用

12组预定tile，真实Down/M64和strided/tail共24配置、72相位artifact通过GPU和独立CPU审核；89388个selected FP64结果、4798764个完整BF16 cast、6个完整format decode相位、原A8量化和W8A8整数控制、guards/RNG/输入均通过，原checkpoint与冻结源不变。

|Down/M64路径|tile|中位数ms|
|---|---|---:|
|同进程BF16控制|实测原生BM64/BN64/BK128/w16/s2|0.058685|
|已有W8A8 NT控制|旧预定配置|0.066933|
|旧W8A16 NT控制|BM64/BN128/BK128/w4/s1|0.226373|
|与原生相同tile/threads/stages的W8A16|BM64/BN64/BK128/w16/s2|0.098242|
|最快W8A16邻点|BM64/BN64/BK256/w16/s1|0.090219|

7轮轮换、每轮20次热Graph，计时包含权重读取/转换/dot/scale，排除一次性CPU打包和诊断。最快候选比同进程旧候选约2.51倍快，仍比BF16多53.7%耗时；不接模型/Math/正式benchmark，不采用。尾部最快仅约0.88%组件变化且非实际目标，不借它宣称模型收益。

该结果支持继续重视tile/线程/流水组合，但未完全解决W8A16转换/计算成本，不能单靠占用率下确定结论。下一独立候选W4A8：压缩存储、解包INT8、现有A8量化、INT8 dot分组FP32缩放，既要计入激活量化，也要新验证每K块实际INT32 partial。不能借W4A16或现用W8A8的Math。

raw `/workspace/benchmark-results/20261022_w8a16_native_tiles_GPU_tianshu`，大数组仅容器。代码、报告、关键JSON/CSV放GitHub，正式发布保持5695.95 total tok/s。
