# 真INT8 GEMM静态权重预转置：NT对NN独立组件

当前合格b9b692c使用W[N,K]，右矩阵访问col*K+k；候选静态存为W[K,N]，访问k*N+col。仅三处可逆源码编辑：一处加载寻址、两处Python wrapper形状。A量化完整AST、整数dot、mask、缩放、输出、ABI和selected tile不变。与早先BF16投影重排不同，不预设收益。

实际layer0 GateUp与LM权重，M1/3/7/8/17/32/37/40/48/56/63/64、strided37以及N130/K259/strided17，共28场景×3次变输入/权重符号/scale的同Graph重放。检查全部有效INT32/BF16逐位、完整GPU整数参考（实际绝对乘积和<2^24）、CPU全部A和checkpoint选中原W重新量化/INT64、全部原/转置W字节、256B对齐前后guard、输入/RNG。小尾部W来自已量化完整bank后截取，scale仍使用完整原K2048，CPU独立按同一真实来源复核。

七轮交替×20热Graph，包含每次完整A动态量化+整数GEMM；静态W打包/转置/上传和数值核验在计时外明确记录。LM8整数只是组件场景；当前模型LM8仍BF16，未来不因此改变模型精度路由。

CPU_GATE只证明三处源码/ABI与既有整数界，不是GPU数值或性能。所有原始PT/编译产物只容器，GitHub保留代码、报告、关键JSON/CSV。GPU与当前KV空分配服务串行；若KV有稳定收益优先新质量/正式4K，本组件待后。无新模型成绩、无完整16K，截止北京时间2026-10-21 23:58。
