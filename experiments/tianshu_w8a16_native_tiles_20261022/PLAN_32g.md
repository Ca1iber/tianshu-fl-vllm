# W8A16按实际原生配置补齐tile/线程/流水

IXKN直接CSV和独立CPU审核确认：Down/M64 W8A16旧候选grid16、4warps、共享49152B、115 vector regs、占用7.30%、TCU16.09%，kernel225.876us；原生BF16 grid32、16warps、2stage、共享65536B、25 vector regs，占用25.53%、kernel56.038us。单独读取真实libtuner best_config确认原生BM64/BN64/BK128/w16/s2。指标显示具体差异，不能把它们单独当成完整因果证明。

本轮只测真实Down/M64和M17/N131/K259 strided/tail（W8A16后缩放、NT）。12组预先固定tile，包括旧4warp/1stage控制、原生16warp/2stage相同tile，以及8/16warp、BK64/128/256、BN32/64/128的邻点；pipeline/parallelism作为整体候选。算法与前版W8A16 NT后缩放不改，原始weight只读。对齐原生tile不代表已取得收益。

完整decode和GPU FP32参考，3次改变同一Graph的输入/weight/scale，独立CPU FP64 selected点积、全部输出cast、所有activation量化和W8A8整数控制，guards/RNG/输入保持。每tile7组轮换，每组20次Graph，含实际读取/转换/dot/scale。一次性CPU打包/诊断不计时。通过组件10%门槛再评估整体收益、全42层新增权重内存和新Math；没有组件收益不跑完整benchmark。

GPU仅此一任务，实际空闲检查，最多30分钟，今天北京时间23:58硬截止；每天06:00–次日00:00授权。源码冻结后不改，失败原件保留。raw只在容器，GitHub保留代码/报告/关键结果。
