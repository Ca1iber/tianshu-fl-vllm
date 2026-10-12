# 天数原生 Decode 启动参数筛选

在已验证Native GQA L0 Prefill、Hybrid和adaptive Decode基础上，独立进程筛选3D Decode启动参数；安装源码不改。

原KV32/w4/stage1长8场景通过，但14个短KV/填充场景全部失败，不能采用。保持原KV16和w4、只将pipeline stage3改stage1后，长8场景及14短场景通过，stage2短场景也全部通过。原kernel body、grid和分块保留。

实际4K/1K、C64/N128三轮排首轮3186.657122 total tok/s，比同规格控制2742.829558提升16.181376%；384请求usage4096/1024均合法，三轮均0新增调优。两测量轮极差/均值5.323760%，这是小规模协议。

新的独立Math-500 Level3为102/105，97.142857%，无请求错误，严格>95%；真实Decode与Native Prefill路由、源码SHA以及全部原始预测/评审已审计并本地SHA备份。质量检查点tianshu-verified-decode-stage1-20261012。

正式原脚本C64/N256四轮、排首轮正在运行；尚无本候选完整正式成绩。当前正式最好仍2712.55 tok/s。`serve_tianshu_minicpm.sh`复现质量通过的候选服务参数；原生Prefill目录保留上一正式最好启动器与标签。不会自动将短测数字写成正式最好。

报告与原始索引位于results/20261012/launch_32g、service_stage1_32g和math_stage1_32g。未运行优化版完整16K；算子中少量长KV数值检查不等于完整16K性能评测。性能剖析TODO。
