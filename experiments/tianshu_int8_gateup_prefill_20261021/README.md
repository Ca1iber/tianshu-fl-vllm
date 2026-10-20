# GateUp prefill 真 INT8 NN 组件（仅准备）

当前模型 prefill M≥128 保留 BF16 linear_contiguous。这里从原真实权重 INT8 探针分叉，只测试 GateUp 的 M128/256/512/1024/2048/4096 与六个事先声明的 NN 分块。沿用已通过小 M 数值门的同一 NN 内核字节，静态 W 以实际 contiguous [K,N] 储存；每次计时包含完整 A 动态量化。共同基线使用当前 prefill 实际 flag_gems.linear(a.contiguous(),w,None)，不沿用早期较慢 mm 基线。

保留三次变 A/权重符号/scale、完整 GPU INT32 参考（实际 sum(abs(products))<2^24 才按精确整数比较）、BF16 量化参考、全部 A 和 CPU 选中 INT64、guard/输入/RNG/Graph 重放检查；同时独立记录相对原 BF16 的误差，不冒称量化等价。末尾 guard 16 个元素，起始 malloc 256B 对齐，权重转置为 view 后 logical stride 不同；所有 GEMM 接收其 contiguous 转置。

此组件与当前模型小测串行，不自动排队启动。需等待 GateUp NN 模型对照完成及真实 GPU 空闲；若模型小测有明确收益，优先质量与正式 4K。组件之后须独立 CPU 复核。任何 prefill 模型接入都会改变量化范围，需要全新 105 题 Math 严格>95%。无新模型成绩。原始文件只容器，代码报告关键结果推 GitHub；23:58 北京时间停止。
