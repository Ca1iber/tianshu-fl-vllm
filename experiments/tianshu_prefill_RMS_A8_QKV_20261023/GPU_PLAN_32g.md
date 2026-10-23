# 首次GPU控制器：完整prefill QKV链，不接入模型

CPU_PREPARE_GATE及CPU_PREPARATION保留CPU准备时的原始状态（当时尚无GPU probe）；旧初始原型在Git 0baa1bb保留。本次补齐probe_gpu.py、verify_cpu.py、run_component.sh，chain.py数值不变，仅把norm编译记录按M和residual分开，分别验证512/4096真实返回kernel与原False/True CUBIN相同；新GPU_INPUTS在首次GPU前覆盖全部实际代码、输入、计划和来源。原CPU输入/父数值kernel/配置不变。

固定18数值case（M512/4096、layers0无残差及20/41有残差、三预定输入阶段），每臂完整独立FP64 norm界与全量A8/scale，全部84,934,656个norm单元；每case每臂8行×16列，4608个完整K2048 INT64 projection参考单元，原INT32及显式FP32乘两scale/BF16舍入精确匹配。明确投影独立参考是抽样，不冒称全矩阵重算。36个真正数值Graph返回对象逐一毒化重放，norm/residual/q/scale/projection全字节保存核验；INT32全矩阵在GPU逐字节比较并记录完整hash，同时保存抽样整数。数值相同的graph/eager文件仅在同试验内硬链接，原始路径、hash和独立实际Graph验证保留。

共24个timing Graph（6形状×2臂×chain4/16），实际捕获调用数记录不当成replay次数；最后完整返回对象与同臂phase0数值结果全字节一致。完整链eager/graph4/graph16，各5轮排首轮，轮内交替臂顺序，warm10/replay20，共180计时行。M4096三bank的eager和graph16完整链降幅全部>3%、各臂chain4/16差距<=10%才component_lead；M512仅诊断。输出guard范围为可控制的q/scale/INT32/projection和候选norm/residual，IR编译器内部norm分配以全部数学参考/真实Graph字节检查，不伪造外部保护区覆盖。

Operator所有临时Tensor在单次调用内分配；原BF16权重与NN pack保留，原native _int_mm及dequant步骤/顺序共享。固定K2048/eps1e-6/4warp1stage，norm与QKV epilogue CUBIN逐M匹配既有组件。对照为独立编译本机IR-native函数，不宣称与完整模型CUBIN完全一致；通过仍需新的实际模型接入、42层路由/编译范围审计、新鲜完整C8/C64各105题严格>95%及同期短4K，不能把组件收益当模型成绩。

本轮RMS+LM组合短4K已完成且未过筛，其27948和条件正式30742都已终止，无正式GPU。GPU入口仍再次拒绝任何原队列活动、其他服务、GPU非空闲；要求>=2.3GiB可用空间、剩余>1200秒，15:58Z超时终止。当前组件尚未运行，首次前SHA冻结；原始数组/缓存/日志只容器，GitHub仅代码、报告与关键门槛。
