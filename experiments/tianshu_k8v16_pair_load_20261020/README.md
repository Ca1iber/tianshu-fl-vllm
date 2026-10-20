# K8/V BF16 成对K读取与16位解码

独立于正式future-KV版本的组件试验，不修改已测模型或运行源。现用7b94...S16 reader为控制，比较scalar16（仅软件8位解码中间uint32→uint16）、pair_rows与pair_columns（每次uint16读取两个K字节，先各自恢复BF16再组合成原K矩阵）。两种pair的矩阵构建方向不同。K缓存仍8位，V仍BF16，不是整数QK，不改量化公式或数值。

此前四字节K打包慢约41%，本轮不预设两字节会好。CPU穷举65536对码值并以独立bytearray/struct验证stride与对齐。GPU使用真实12KiB页、现用writer/Producer/Reducer，12场景×4路×两相位，完整FP32/CPU参考、变数据同Graph、缓存/边界/RNG与逐位对照。九轮交替Graph计时，每组20重放。B64真实编译产物只留容器。

先完成组件再决定是否进入4K N64服务；没有足够收益即记录停止该方向。组件通过不代表Math或端到端成绩。受2026-10-20 23:58北京时间硬截止约束。
