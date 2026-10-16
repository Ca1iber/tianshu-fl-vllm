# 天数Scheduler/KV只读诊断

当前GEO/M2048/.cv/KV16/S8最佳算子，原serve参数、75warm、budget2048、maxseq256及async保持。包装仅转发原方法，记录调用；不改请求队列、KV页长、分配或调度决策。V1漏转发throttle_prefills参数首请求失败，失败包单独保存；V2核验实际签名并测试默认/位置/关键字参数后完成运行。

2轮4K/1K C64 N64，128响应全部成功，固定种子canary一致，额外16输入/1输出sentinel用于刷盘。第1轮3845.467174tok/s、TTFT16432.313653ms，第2轮4033.915243tok/s、TTFT12730.526684ms。这是带插桩诊断数据，不作为新模型性能成绩。

75份冻结源码及实际39环境、全部响应和2349步JSONL本地复核。2076步纯decode、256步混合、7步纯prefill、10步空调度；B64纯decode1790步，实际throttle_prefills为0步。GPU KV剩余块最少12413，分配失败0。

带插桩B64纯decode的原schedule wall中位数1.3802725ms、内部allocate_slots汇总中位数0.7715755ms。两者是父子嵌套，不能相加；方法内包含下层observer的开销，不能作为无插桩CPU时间或GPU占比。没有据此定量宣称CPU/GPU重叠导致上一项融合无收益。

调度内allocate_slots共131361次，122786次无新块（93.472187%），共40972个新块。SingleType cache_blocks同样131361次，其中122910次已有完整块早退（93.566584%）。调度外Async输出提交还有131105次manager cache_blocks，其底层131105次均早退。说明有可精简的CPU调用，但其端到端收益仍需实验，不能只看调用数采用。已有早退并非本次新实现。

另独立读全部step：259步进入时有waiting，含3个canary/sentinel初始步；其余256个吞吐请求调度步全部用满2048token预算。此次未见明显调度预算空耗或KV分配失败，暂不盲调chunk/budget。新的NHD/HND物理KV布局组件验证另建目录。

原件/workspace/benchmark-results/20261016_scheduler_kv_trace_v2_tianshu，4974963bytes，SHA08a3e11bb6d9f248479d2137b99e30df8b757dca4f78646fcceb9dd35ba251d1。复核：`python3 experiments/tianshu_scheduler_trace_v2_20261016/verify_downloaded.py --root <下载目录>`。

本诊断不采用任何新优化、不运行新Math/正式/16K，正式最佳4093.81/Math102保持。
