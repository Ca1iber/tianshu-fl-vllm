# NN GateUp单dot：两段显式load，join交错列，直接split

原32组single_dot按分段列地址load+transpose/split在M8给出错误Up整数，源码与完整debug98304 CPU INT64证据保留。max_contiguous BN不修复、binary未变；tl.cat默认写法编译不支持。没有降低容差或把失败改通过。

新实现分别按简单连续列load Gate/Up两个BK×BN片，再tl.join成BK×BN×2并reshape为BK×2BN；dot后直接reshape BM×BN×2、tl.split，不使用不连续的单load或tl.cat/trans。权重仍原NN物理布局，无额外W副本；只是寄存器内配对排列。恢复原BF16投影舍入、调用原FlagGems标量。CPU证明交错列顺序与全部12288列各一次，非single分支AST严格等于原two_dot。实际是否正确依赖GPU完整整数和投影检查，不能由CPU布局证明代替。

actualM8/64各连续/stride2055共4形状×bn64/128×warps2/4=16配置，bk128/stages1，不补实际M。共同对照使用当前选优NN GEMM加声明的FlagGems SiLU，量化包含、静态W转置排除。每组3次变A/W/scales同Graph，计时版完整输出逐位相同、诊断版全INT32/BF16投影相同，checkpoint原行独立CPU INT64、全A量化、激活容差、完整输入/权重/RNG/guards全部保留；7轮交替×20replay。保存完整计时输出离线重查，若首次失败先保存完整A/Q/W/scales/native projection及双方输出，再拒绝。

新组件不重跑原失败控制，不声称原32组已完成；parallel槽不用，不并行GPU。当前two_dot独立完整16组另在tianshu_nn_silu_two_dot_20261021跑，不是本次数据。FlagGems标量组件不自动等于模型Torch编译激活；模型FX接入/新Math另审。原NN capture验证将需真实融合调用证据，不能伪造旧NN launch过断言。

1800秒与23:58硬停，源码先冻结，后独立CPU。全部raw仅容器，代码报告关键gate进GitHub，无本地副本。正式5586.28/Math102of105不变。
