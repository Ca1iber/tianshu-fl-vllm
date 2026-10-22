# LM动态批次补齐64与FP32输出融合

独立组件，固定M16/32/48/63/64，真实原LM权重130560x2048和原W8 pack。基准M64是实际r3 W8A8 Triton BF16+to(FP32)，其他M是真实UnquantizedEmbeddingMethod.apply BF16+to(FP32)，启用与WorkerFL相同空黑白名单FlagGems注册并记录实际vendor linear调用源，不把torch.mm当成实际基准。

候选A8量化时将实际M行后补零到64，M作为do_not_specialize运行时参数，所有64行q/scale均写出，不读越界输入。固定原M64 GEMM配置，将GEMM结果显式BF16舍入后直接FP32写出，返回前M行view，不隐藏全局workspace。M64与原路径逐位一致；M<64相对原BF16是新量化，不能沿用原正确率。组件不拦截模型，未来模型默认保留M<=8 BF16。

每个M三组独立BF16输入seed202610242100+100*M+phase，首行0，并改变权重符号/scale。oracle为原量化/原Triton完整64行INT32和BF16：新量化q/scale全部精确、全部INT32精确、所有64x130560个FP32输出逐元素等于原BF16转FP32，包括补零行。保存实际图返回对象，污染输出/q/scale后图重放，分别核验eager与graph返回，输入/权重/RNG/哨兵不变。独立CPU每阶段固定8行72列完整K INT64与顺序FP32缩放/BF16 RNE；原BF16基准对FP64乘积按gamma(2K)绝对点积误差加一BF16 ULP核验。该基准误差界不授权新量化质量。

计时前恢复原权重/scale，图/eager各七轮、首轮排除、后六轮中位数、每项20次、交错顺序。按实际LM64 eager路径筛选：预声明cutoff32、48、64，选择其以上所有已测M点均比实际基准快>5%的最小cutoff；M16仅观察。非采样点不冒称已测，所有range仍需完整模型实测。只有cutoff<64才考虑新动态批次版本，仍需新完整105题C8/C64>95%、短4K>0.5%与稳定性/TTFT门槛，再条件正式4K。组件误差不代表模型质量。

原903源/pack签名/checkpoint前后哈希冻结，代码/计划在首次GPU前冻结，失败修正另建版本。上一GPU与CPU pipeline必须终结，当前GPU空闲才执行。最多15分钟，不越过2026-10-22T15:58Z，GPU串行，数组/完整日志只容器。
