# 天数原生 Decode 启动参数筛选

旧联合扫描修改了分段与KV tile，没有覆盖warp数和流水线级数。该实验在当前adaptive分段策略下，保留原始M16/BLOCK_Q2网格和完整attention算术，仅在独立进程的kernel launcher选择KV tile16/32/64/128、warp4/8/16与stage1/2。原始启动配置作为控制，记录编译器实际metadata，不假设Triton默认stage值。

8个场景：B1/8/16/32/64、4K上下文，B64/ragged 5K，B16/ragged 16K和B2/ragged 32K。真实K/V交错分页、随机physical page与非连续Query/KV；全量FP32、选定CPU、原生对照和输出/scratch尾部验证，轮转CUDA Graph事件计时。14个候选加原始控制；共享内存等编译失败单独保存，不能把失败组合当数值通过。

这只是算子微测，不是完整16K benchmark。GPU尚未运行；尚无数值、实际服务、质量或正式成绩收益结论。安装源码保持冻结，串行等待先前融合数值实验清理GPU后运行，动态读取最新预约截止时间。
