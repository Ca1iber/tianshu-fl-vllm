# 天数静态INT8 KV启动配置与寄存器资源

此前INT8固定沿BF16的tile16/w1/S8时，scale外提仍慢6.37%，producer158 regs而BF16为85，均无spill；这只是资源线索，尚未证明瓶颈原因。本版保持同一数学和scale外提helper，预设8项warp/tile/segment组合，主要测试tile16/w2和w4能否降低每线程工作集，另列warp8、tile32/w4、S4/S16。没有改producer算术或位宽、cache.cv/page16，独立BF16当前配置作控制。

仅B64、4K和5K两场景；两阶段改变Q/K/V并清空输出Graph，对自身量化FP32参考与CPU抽点检查，BF16近似误差单列；检查output/segment尾部，7组轮换事件。无量化writer/真实模型框架集成、质量或吞吐结论。若无组件收益不进服务；若有明确收益仍需实现所有真实KV流程与校准，并新Math105>95。

当前只准备源码，待调度服务释放GPU后再决定是否运行；不与服务争用GPU。无16K、旧冻结目录/权重/安装/最佳启动器/原benchmark/沐曦改动，23:58硬停。

本版手工构造页内K/V交错的两种dtype storage，K/V view的元素stride均8192/256/128/1，对应BF16页16KiB、INT8页8KiB。匹配页内排列但未接入完整模型allocator。编译错误或资源不足单独保存失败配置，其它配置继续；数值或运行时错误仍使实验失败，不靠排除数值失败获得通过。
