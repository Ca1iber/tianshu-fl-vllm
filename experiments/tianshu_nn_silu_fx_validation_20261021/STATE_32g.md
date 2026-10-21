2026-10-21 17:34 北京时间

# 当前任务顺序（最新用户指示优先）

用户刚明确：先把本轮质量/正式4K测完，再按各环节和M范围探索重排/融合，生成分角色的范围白名单。新方向准备目录experiments/tianshu_m_range_stages_20261021：PLAN、两份可变M BF16 packed移植、通用INT8融合、SOURCE_PREPARATION与真实warm/observed M摘要。只CPU准备，GPU driver还未写完，未冻结完整GPU入口，不能宣称已实测或直接启动。

已核对沐曦只读bare参考：flagos-contest ed3bcfe的_BUNDLE_ROWS五点576/1152/1664/1920/2048是MLP和QKV/O共用；LM另有1/8/32/64。ops来源8eb57f17。适配只改vendor guard，AST确认所有设备计算函数不变；INT8 fused拷贝只扩host M/BM/warps并禁止已知错误single_dot。实际天数GateUp prefill为部分M INT8，必须和真实INT8基准比，而非仅BF16慢对照。QKV/O/Down/GateUp/LM分开，记录W静态重排bytes和A逐次重排/量化成本，再确认42层轮换与4K服务。计划包含范围加密/边界/真实可达点，不用两个端点推整段。

先前Q2准备40efaa4 **暂不运行GPU**。复核已有multiquery_paths/recovery完整26场景覆盖Q2/T16/W1及gather/bytefallback，准备有重复；新Q2默认arena32768也不是当前SmallArena8192的等价对照。REVIEW_32g.md已如实记录，原准备源和CPU gate不修改，不冒充GPU通过。用户M范围方向优先，后续不要跟着旧STATE运行Q2。

# 正在运行

C64质量/正式4K串行watcher PID249716，phase=/workspace/benchmark-results/20261021_nn_silu_fx_c64_quality_runner_tianshu/phase.txt。当前C64_math，质量controller250826、API250828、Engine250980、EvalScope251374，17:33实查均live，103/105完成、剩余长答案正常生成。root=/workspace/benchmark-results/20261021_nn_silu_fx_c64_quality_tianshu。M64真实FULL图已重放1710次，42融合capture route存在。必须等105题最终>95%及完整冻结审核；不因观察超时重启、不重复抽分。

入口2706c9e冻结：C64补充全部105题/max32768，同450数值源，仅FutureKV candidate→control准入，保留原生allocator，实际42层M64融合replay>=100与LM64 INT8>=10。gate=experiments/tianshu_nn_silu_fx_c64_quality_20261021/C64_QUALITY_GATE_32g.json。通过后watcher自动正式N256 C64 4K/1K四轮排首轮，root=/workspace/benchmark-results/20261021_nn_silu_fx_official4k_tianshu。每完整benchmark轮完成向用户报吞吐/TTFT。质量失败不重跑；仅审核metadata错则保留failed、CPU新目录恢复，不重复GPU补metadata或捏造覆盖。所有已启动py/sh/CPU gate/PLAN保持冻结。

官方Math8已完成并全runtime/Math审核：101/105=96.190476%，105成功/0错误，104stop+1max32768。raw=/workspace/benchmark-results/20261021_nn_silu_fx_math8_tianshu；runtime450源/62环境/13原命令。关键86KB审核在tianshu_nn_silu_fx_math8_results_20261021/MATH_GATE_32g.json，9aedfd6；完整4.5MB重复逐层记录仅容器并有SHA。原runner247467=Z，API/EvalScope已消失。普通Math8不能验证M64融合实际请求，故有C64补充。

三臂小测：原始407参考5508.9690，静态64对照5529.4693，融合5554.9744 total tok/s，整体+.83510%，576请求全部成功，零就绪后调优。新正式尚未完成，正式最好仍407源5586.28 total/1117.25 output、TTFT3780.63ms、Math102/105。不运行例行16K。

截止2026-10-21 23:58北京时间/15:58UTC（预约到10-22零点），目标继续，不自行暂停/完成。SSH master=/tmp/tianshu-20261021-afternoon-control；GitHub分支exp/tianshu-v-int8-20261020，已push7880da9，下一步push本次准备和STATE。所有代码/报告/关键结果仅容器和GitHub，原始大日志张量容器，不留本地。两旧GateUp用户dirty文件和旧CPU logs不动，不改沐曦参考。
