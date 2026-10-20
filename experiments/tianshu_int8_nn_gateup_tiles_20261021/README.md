# GateUp NN 分块有限搜索

沿用已通过 28 场景 / 84 阶段完整数值检查的相同 NT 与 NN 内核字节，只改变 NN 的 tile/warps/stages。真实 layer0 GateUp、M8/M64，每个 M 的共同对照始终是当前模型使用的原 NT tile。17 个预声明候选，不动态无限调优。每候选三个变输入/权重符号/scale 阶段要求完整 INT32 与 BF16 逐位、完整 A 量化、CPU 独立选中 INT64、完整权重/输入/RNG、256B 对齐前后 guard；编译资源错误独立保留，不计通过。7 轮交替次序 × 40 Graph，含 A 动态量化与 GEMM，静态 W 转置排除。

这是组件，未运行模型、Math、正式 benchmark。先完成再独立 CPU 复算选中原 checkpoint W、数值与计时。原始 PT 与编译/快照留容器；代码、报告、关键 JSON/CSV 上传天数 GitHub。截止北京时间 2026-10-21 23:58。
