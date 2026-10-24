# 待实现：当前r5量化KV消费者的BM8/direct候选

当前r5实际成对K读取为f1df...，与21日BM8 v2/v3父源完全一致；当前997来源/65环境和实际capture未包含BM8。复用已通过8044544元素bitwise组件资格的BM8 direct及v3 M64-only选择器，只替换IS_3D/N64的query tile16→8、BLOCK_Q2→1、grid96→64，保持K8/V BF16格式、writer、softmax/reducer、分段4、所有W4/W8/RMS/采样及checkpoint。其它M/2D委托原r5。

本目录目前只有计划，没有新模型源码/CPU_GATE、GPU运行或新组合质量。旧短N64+2.4325%与C64 103/105仅为研究依据，旧C8/正式缺失，不能借给r5组合。下一步须新candidate_profile：两模式均先parent_r5.prepare(root,'candidate')，避免parent control关闭QKVO W4而错用r4对照。新增policy应只在candidate装BM8，control保持r5；同一套源与全部既有payload，源快照大SDK仍引用旧独立捕获副本，严禁重建/覆盖该SHM根。

当前工作盘约125MiB，所有新profile/cache/大输出优先SHM；不复制3份大SDK库。若需要新Inductor缓存，候选自己记录TORCHINDUCTOR_CACHE_DIR指向SHM，不能改已发布r5。新组件/机器码若实际flags不匹配旧canonical签名，先独立验证当前环境，不绕过审核。

须新增最终JIT/source/grid/BM/机器码审计，不能用子选择器之前的pair记录充当实际新kernel证据；绑定真实uniform64 Graph capture和运行replay。新C8/C64完整105题>95%，无截断/漏题；同场freshN128每轮不同seed、两组对应seed相同、prefix hit≤1%、零抢占/就绪后autotune，无计时期间CPU准备。达到预定吞吐/TTFT/稳定门槛后才原始N256×4排首轮4K。用户未要求常规16K。保持报告/Git和23:58BJT停止边界。当前r5/tag/正式结果不改。
