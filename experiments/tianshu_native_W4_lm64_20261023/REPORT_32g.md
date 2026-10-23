# 真实LM64 W4组件：dotsaddz相对已启用W8A8核心快17.61%

当前已启用对照为b9b692原始自写dense.quantize + dense.mm，bm64/bn128/bk128/w4/s1，实际编译93寄存器/0spill/24576shared。它不是前版未采用的原生_int_mm/dequant，也不是BF16对照。本实验的对照复用预分配q/scale/output，只比核心，不把全部模型Python调用开销计入；候选含原生输出分配，此设定偏向对照。

排除首轮、其余6轮中位数，每轮20次。eager：当前0.611353254 ms；AWQ0.535017300 ms（+14.2679%，省0.076336ms）；dotsaddz0.519834042 ms（+17.6055%，省0.091519ms）。graph：当前0.615958858 ms；AWQ0.536821628 ms（+14.7418%）；dotsaddz0.521622419 ms（+18.0852%）。原r3正式记录中LM64的15385次Python调用均在主模型图外，因此eager核心更贴近当前调度；仍不能直接推导整机收益。

权重是真实lm_head.weight [130560,2048]，W8打包/scale/原BF16全SHA匹配已通过的模型签名；267386880个逻辑W4值的两种完整pack逆变换通过。3个独立确定性M64 fixture（随机输入，含零行，非线上hidden-state trace）；两W4候选的50135040输出格完整FP64检查通过，kernel相对量化BF16反量化权重参考的L2约0.1657%，逐格均在首GPU前冻结的阈值内。W8核心A8/scales完整独立重算精确，98304个全K INT64/BF16输出精确。所有9组完整输出的eager/graph poison逐字节一致、有限，全部输入/权重/scale/zero完整字节、版本、guard、RNG、checkpoint和冻结来源不变。

RTN W4相对原始BF16投影的完整输出L2约12.97%–13.20%，当前W8约1.37%–1.39%。两W4后端输出一致。这是运行时group128 RTN加已验证的不同物理pack，并未做激活感知校准。需要模型Math500，组件通过不等于模型正确率通过。

冻结aff9ad6a87543b7265e723b18a4be0959854250e；原始CPU/GPU目录 `/workspace/benchmark-results/20261023_native_W4_lm64_{CPU,GPU}_tianshu`。GPU结果SHA：25b3cd4152a6928b489349431f2559a1bad8440e1c1667bd75738ea87dd3699f。代码、关键计时、门禁、报告进天数GitHub；大数组只在容器。没有改变原始checkpoint、模型架构、已合格服务，没有正式benchmark，最佳仍为r3 5950.75 total tok/s。

优先集成绝对节省更大的Down64（图省0.3804ms），先单独质量验证；本LM64候选保留待后续，避免第一次混合两个精度变化而无法归因。
