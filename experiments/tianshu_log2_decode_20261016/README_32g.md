# 天数：Decode在线softmax保持log2域

新独立副本基于当前cv源8deb及实际helpers源59d9d34b。只在受守卫的普通3D Decode下把score scale乘log2(e)，循环内softmax_step用exp2，分段M写回前乘ln(2)，保留原自然指数reducer契约。原tile/warp/stage/grid/分段、.cv、L0、int64及指针都保留；guard不支持时仍原始计算。LOG2_DECODE=False生成副本作为恒等控制。

这是浮点舍入变化，不能声称逐位等价或复用旧Math。CPU35个在线softmax/自然域分段合并参考检查通过（double误差<1e-10），生成源码精确逆变换、整数cast数不变及实际CLI help通过。GPU21场景42候选已完成并通过独立FP32/CPU/变QK Graph/尾部检查；C64/4K+1.51451%、5K+1.45610%，不采用、不做服务或新Math。候选舍入与原版不同，不声称逐位等价。见[REPORT_32g.md](results/20261016/micro_32g/REPORT_32g.md)。

计划21场景/42候选：17个既有4K/5K/短边界/stride/padding场景，加B64-4K query×4、B8短query×8、低幅度×.25、全零query。Graph变输入改为query×.5、K取负、V×.5，让QK分数也变化，而不只是旧Q/K同时取负保持分数。保留GPU FP32、独立CPU抽点、尾部、原始9轮交替event和实际编译产物。GPU父子串行、有17:58local硬截止，冻结11源码。

运行：`python3 experiments/tianshu_log2_decode_20261016/run_probe.py --output NEWROOT --stop-at-utc ISO`。有明确局部收益后再做真实服务对照，新Math>95%后才正式；当前正式3992.97/Math103/105不改。
