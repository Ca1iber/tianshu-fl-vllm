# CoreX 原生 INT8 GateUp 完整路径对照

支持性V2已在M64/K2048和K6144、NN/NT布局、3变化相位完成全CPU INT64及Graph out核验。新假设：torch._int_mm原生整数GEMM可能比当前Triton整数融合更快，足以覆盖额外INT32中间写读和独立epilogue。

使用实际第0层Gate/Up checkpoint权重与既有W8签名；M64，连续/stride2055两种A。当前控制是正式采用的tianshu_nn_silu_two_dot kernel，BM64 BN64 BK128 W4 S1 two_dot，量化仍旧NN逐行4warp。候选为同一量化+torch._int_mm(out)+Triton epilogue，精确保留BF16投影舍入、FP32 scale乘法顺序和同一FlagGems SiLU scalar。两种静态权重布局NN/NT，6种epilogue BLOCK256/512/1024×W4/8，全部计入热Graph；CPU一次性权重打包和诊断不计时。

3相位改变A/W码和scale，完整INT32与现用诊断结果一致；完整BF16投影/激活输出逐位一致，再独立CPU从checkpoint重建量化码、全A8和scale、selected INT64 dot、完整投影舍入、selected FP64 SiLU近似参考。原模型SHA前后不变。guards、输入/权重/RNG、复用同Graph输入更新、实际当前控制binary签名检查必须通过。7轮轮换×20 Graph replay；预定12候选×2strides，保留失败，探索选优不当作holdout。

本组件不改现用路径，不声称模型精度/速度。原生INT32 workspace为M64×12288×4=3145728字节，每种布局静态W8为25165824字节；不能直接推断引擎净显存开销，若进入服务还需原生DeviceMemoryProfiler、bank清单、fresh C8/C64 Math及4K小样本同期对照，阳性才正式benchmark。M8不支持native接口，维持原实现，不补齐隐瞒开销。

完整冻结源/计划后GPU，GPU空闲先检查；最长30分钟、北京时间23:58硬停，每天06:00–24:00授权。不与A8 geometry GPU并发。代码/关键结果/报告天数GitHub，raw仅容器，不创建本地项目副本。
