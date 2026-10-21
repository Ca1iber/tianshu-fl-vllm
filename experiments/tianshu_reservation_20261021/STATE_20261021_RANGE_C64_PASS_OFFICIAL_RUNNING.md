# 2026-10-21 20:57北京：范围融合真实C64通过，正式4K启动

当前正式推荐仍5618.01 total tok/s /1123.60 output，TTFT3835.46ms；旧450-source tag不变。
逐M GateUp融合23个正收益M，保留M256原路径。相同471源N64短测5628.4467 vs5557.7773（+1.27154%），TTFT -4.10248%。
原Math8全105题103分，但新融合实际0层而审计失败；原GPU result保持audit_failed。独立CPU复核在tianshu_m_range_math8_coverage_recovery_20261021，不重抽题。
新C64质量101/105=96.1905%，实际M128/768/4096×42层融合、C64旧融合图重放及LM64 INT8都通过。
续接控制器tianshu_m_range_prefill_resume_20261021，runner288111；正式原N256四轮排第一轮，正在编译/预热。最晚23:58北京GPU硬停。
串行队列：BF16大M1008项（runner288549），完成父质量/正式后首次运行；再QKV小M物理NN布局34项（runner290252）。不能并行GPU。
QKV新组件是decode M8/64的NN/NT/BF16三臂，有别于此前NT prefill已测+.293%未采用；每相位完整NN整数输出都将由原checkpoint独立CPU重算。
所有源码、报告和关键结果放GitHub和天数容器；历史26项raw生成源均有现有Git release bundle的逐字节副本。主机/tmp仅SSH控制socket/FIFO，不复制项目内容。
