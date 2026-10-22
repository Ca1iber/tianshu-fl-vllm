# W4A16 每组点积后缩放

在成对读取/交织的相同 packed 格式上，每K128组的整数权重转BF16，BF16 activation 与该整数矩阵点积得到 FP32 partial，再乘本组每输出通道 scale，累加 FP32。避免每个 weight 的 q*scale 和 BF16 round；不是原生 INT4 dot，数值顺序有改变，旧 Math 不借给此路径。

两种W4格式，各测tail M17/N131/K259和真实Down/LM head M8/M64，三组固定tile，三相位变更权重/scale/输入，7×20 Graph 对照 BF16/W8A8。数据打包、单独的完整 decode 和全量 FP32 参考不计入候选Graph；实际读取、解包、所有 dot/scale 都在计时图内。图的输出必须通过新算子参考，不以 BF16 rounded 权重矩阵代替实际算子。

独立 CPU 从只读原checkpoint重建整数码/FP32 scale，核验完整格式decode哈希、全部实际输出BF16 cast与selected FP64点积；真实GPU FP32全输出参考用未做BF16 round的分组反量化矩阵。误差门槛及guard/RNG/输入不变要求沿用，不能证明模型质量。阳性结果才考虑引擎内存计账/新MATH；阴性结果不跑正式benchmark。

前任务结束、实际GPU空闲再启动。每天北京时间06:00–次日00:00，本日23:58硬截止，单任务30分钟硬上限。独立目录/源码/原始记录，原版本不修改，raw仅容器，代码报告关键结果推天数GitHub。
