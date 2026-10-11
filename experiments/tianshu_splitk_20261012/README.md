# 天数NT BF16 decode GEMM实验（排队待测）

假设：小M且大K的投影采用split-K可能提高并行度。生产默认Iluvatar mm候选的split_k列表只有1。本原型将各段结果存为FP32，再单独归约为BF16，避免以BF16 atomic_add累加；不量化，不复制或重排权重。多一个kernel及中间张量可能使其变慢，收益待测。

M64和M8，qkv(N2560/K2048)、o_proj(N2048/K2048)、gate_up(N12288/K2048)、down(N2048/K6144)，每形状15配置，合计120组。固定输入，对照实际FlagGems NT mm，完整GPU FP32参考和选定8行/32列独立CPU FP32参考；阈值固定atol0.02/rtol0.01，记录失败和源SHA。数值通过的候选预热后各捕获20次调用的Graph，七轮交替回放取中位数，包含FP32归约，不把算子收益写成服务收益。

候选只在独立实验进程使用，GPU任务串行；最多30分钟并受租期11:58截止约束。通过数值/服务/独立官方数学>95%才能考虑正式4K。原始运行目录为`/workspace/benchmark-results/20261012_next_diagnostics_tianshu/splitk`，当前未修改生产安装文件。
