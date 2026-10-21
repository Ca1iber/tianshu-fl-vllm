# 逐M融合进入质量后：下一方向的证据核对

新4K短测完整通过，候选5628.45 vs同源5557.78，总吞吐+1.2715%、TTFT -4.1025%，热轮差幅.2798%（对照.0603%），384请求全成功、0现场autotune/0preemption。正式推荐仍为5618.01版本，新Math8正在跑，之后真实C64补充质量及原N256正式4K；没有新质量/正式成绩前不能替换推荐。screen报告3153b96已推送。

已直接读取当前candidate raw result实际pair_load_record：C64 f1df731b成对K reader真实启动为T16/W1/stage1/S4、grid[96,2,4]、176regs/0spill。旧继承的BF16 KV16 S8核验字段属于参考路径，不可将它误作当前K8 reader的S8。小型当前launch提取另存同名JSON；源文件、分段与KV格式未改。

重新设想T32/T64后，发现现成tianshu_pair_tile_sweep_20261021已经针对同一f1df源码完整测过。112完整GPU FP32输出、304独立CPU向量、真实KV writer和动态同Graph数据/边界/RNG全部通过，当前4K .552935ms，T32/W1 .872142、T32/W2 .858356、T64/W2 1.851001，全部负收益；不能把它们当新方向再跑。T16七组warp/stage扫描也已经全负收益。

安装版Triton iluvatar/compiler.py min_dot_size允许小M/N，但BF16 dot收缩维K最小16。直接把attention上下文tile降到8，会使PV收缩维变8；需要显式填充或另改PV实现，不能仅传TILE_SIZE8假定可编译。占用率可能改善只是推测；没有新GPU结果，更不能把寄存器数减少当速度提升。较大的上下文tile本轮不重复。

优先GPU后备仍为已冻结但未GPU运行的BF16大M：12M×14tile/group×3role×2phase=1008相位，增补group1/8及BM256/BN256真实2048 tile。保留O/Down M2048当前实际特殊tile；同tile比较W pack才归因布局收益。CPU独立fixture/二进制/时序核验器在tianshu_m_range_bf16_large_results_20261021。原始数据只在容器，关键代码/报告上GitHub。须等当前质量和正式控制器终态并确认GPU空闲后执行；失败则先诊断当前版本，23:58北京硬停。没有启动新GPU实验。
