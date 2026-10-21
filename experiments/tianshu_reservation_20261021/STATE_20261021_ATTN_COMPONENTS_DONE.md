# 2026-10-21 11:15北京：QKV/O组件完整结束，GPU空闲

Goal active优化到10-22零点，23:58/15:58Z硬停。上一轮属于progress：新增真实权重组件/完整CPU复核与历史路线核对，未修改推荐版。当前best407-source/62env正式4K5586.28total/1117.25output、TTFT3780.63、Math102/105；tagtianshu-original-command-4k-prefill-int8-20261021。新8192formal5608.40未采用；无当前16K，旧3885.91不同profile。

原双投影PID210196terminalZ、raw20261021_int8_attention_projections_GPU_tianshu整体failed，O8/index4/phase2原BF16参考断言，旧source/CPUgate冻结。QKV9×20完整组件全部独立CPU复核（540selectedINT64/50780160A值）；结果tianshu_int8_qkv_partial_results_20261021明确原run仍failed与QKV-only scope，不重复GPU。主导M4096原BF16mm .492037535→INT8 .428115940ms，+14.9309%局部逆耗时，M8/64/129负收益。当前GUP BF16linear策略仅GateUp，不涉及QKVO；O2048原专用tile保留。

新O独立PID216278terminal，raw20261021_int8_o_projection_diagnostic_GPU_tianshu complete，9×20/540/50780160CPU值全部候选INT32/量化输出通过。CPU原件20261021_int8_o_projection_diagnostic_CPU_tianshu，结果代码tianshu_int8_o_projection_results_20261021。M4096 .387642694→.355951166ms，+8.9033%局部，M8/64/128/129负收益。FP32参考原门槛未放宽，1080原BF16/linear选定检查3native false，最大误差1.154/超阈1.1322，keyGate保留all_baseline_selected_checks_passed=false。32次同一O8原phase0完整输出全部相同/原门槛过，CPU完整FP64 maxerr .0140999；无法解释后续3项或父失败瞬间，不能说已定根因。所有完整PT仅容器。

另CPU复核旧ngram：GPU原生最后轮accept14767/50723≈29.11%；后来248点精确请求数mixed图已实现，旧CPUngram服务+6.51%但3093.85仍慢于旧非Spec4037。旧报告的“下一步”不当新方向，原API响应token_ids=null不能重分词冒充精确token replay。reviewtianshu_spec_history_review_20261021已推f516fda。

下一行动具体见NEXT_ATTN_TOPP_PLAN_20261021.md：可优先新API当前407父profile+旧已有+.807%TopP16K/grid2，并扩B8..64确保C8实际新数值资格，完整mask/RNG/lifecycle/fallback。不是重复已用warp16或无收益online-sum；当前仍GEO32K，oldgrid未采用。API新目录freeze再service新目录。之后按QKV/O absolute收益决定是否组合做N64小4K3+3，再新Math>95与正式N256，不常规16K。新增QKV/O权重真实DeviceMemoryProfiler计费，临时与TopP64行scratch在memory_profiling_fl原上下文yield前warm，不reset峰值；parent audit保留全部407检查。

GPU当前68MiB/no processes；没有服务benchmark后台。未启动新Math/正式。两处用户dirty gateup_silu代码不动、Ops4cdc3b7不变，无local副本、无subagent。SSHmaster/tmp/tianshu-20261021-control04:27建立，16:27前刷新。长保存计划自动审批超时未执行，已用短命令成功写计划，无待用户审批项；不因此重启作业。
