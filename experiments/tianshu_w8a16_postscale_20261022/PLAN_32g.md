# W8A16 先点积后缩放

真实第0层 Down / LM head 的 M8/M64 及 strided/tail case，固定三 tile，7×20 Graph 计时。只用每通道 W8，不量化 activation；INT8 整数权重直接转 BF16（所有 int8 值可精确表示），BF16 dot 后 FP32 乘每输出通道 scale。与前一版的逐权重乘 scale 并 BF16 round 不同，这是新的数值路径，不能借用已有 Math 或 BF16 重建误差证明模型质量。

对照同进程 BF16 mm_out 和已有 W8A8，两者保持原计算。打包开销单独排除/计账；捕获包含实际读取和转换，诊断 FP32 store 不计时。量化格式的完整 decode 仍逐位检查；GEMM 参考改为 A×整数权重后 FP32 scale，与实际顺序匹配。独立 CPU 重新读取 checkpoint，审核全部输出 cast 和 selected FP64 点积；不以 BF16 权重重建矩阵替代实际算子参考。

仅在前一 GPU 作业结束并确认实际 GPU 空闲后运行。原文件、失败记录保留。每天北京时间06:00–次日00:00，今天23:58硬停止，单作业最多30分钟。无组件收益不接模型/Math/完整benchmark。GPU 原始 artifacts 仅容器，代码、报告、关键结果放天数 GitHub。
