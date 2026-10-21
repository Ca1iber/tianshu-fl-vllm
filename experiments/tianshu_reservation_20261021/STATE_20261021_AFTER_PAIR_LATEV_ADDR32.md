# 14:24 成对 K 读核的延后 V / INT32 四路拒绝

预约 2026-10-22 00:00 止，GPU 23:58 硬停；正式推荐仍为 521c154 / tag tianshu-original-command-4k-prefill-int8-20261021，407 源/62 env/原 13 argv，4K 总 5586.28、输出 1117.25 tok/s、TTFT 3780.63ms，Math102/105。新正式成绩没有产生，不把组件耗时当模型吞吐。

刚完成 experiments/tianshu_pair_latev_addr32_20261021。准备提交1b2bc0f已推送。current/addr32/late_v/late_v_addr32，父正式读核f1df逐字代码副本追踪；仅n25..64合法配置启用，T16/w1/s1和原S未变。CPU2816选择覆盖、1760容量、84共同回退、24576独立字节地址；GPU112完整路径、8044544元素重新加载、304独立CPU参考/1216记录参考、504时序组，全输出与current逐位相同。所有源冻结且结束哈希通过；GPU空闲，runner236962终态Z，phase complete。

B64/4K current .552421904ms；addr32 .566245747，late_v .595882463，combined .604182673，均负收益；5K与40/16K组件也负。regs176→167/155/146全0spill。拒绝全部候选，不服务/Math/正式。REPORT_32g/GPU_GATE/GPU_ROUNDS已准备本次推送。

前一轮T16七路全部负收益，a81ad03报告已推送。QKV prefill服务+.293182%未达事前>.5%阈值，6c4775e闭环，exact407参考GPU未启动。INT8 PV全部负，44298f3闭环且独立CPU恢复只改元数据断言；TopP16K质量99/105失败不复测；DownSplitK、grid-only、prefillbound、预算8192等均未采用。

最近以算子/访存为主，接下来应优先查看真实服务执行和框架开销，避免继续以寄存器少推断收益。已知沐曦K累计长度缓存不能直搬：天数builder已共享Qcu，真实paged FlagGems路径没有每层K cumsum/D2H，只有empty_like dummy；FULL_DECODE图内Python分配不是正常replay时的CPU热点。若做新profile，单GPU独立目录、CPU准备和冻结后启动，先短4K，按真实结果挑选优化；不因本文件文字而认为profile已运行。

所有代码结果仅容器+GitHub。本地不留模型/参考副本/日志；user dirty gateup_silu两文件不碰；FlagGems clean4cdc3b7，沐曦只读，不启用子agent。SSH master /tmp/tianshu-20261021-control，16:27前应刷新，建立新连接成功再关闭旧连接。raw只/workspace/benchmark-results。不要修改已运行实验源码；CPU fixture初次失败发生在GPU前，v2通过，原失败日志保留。
