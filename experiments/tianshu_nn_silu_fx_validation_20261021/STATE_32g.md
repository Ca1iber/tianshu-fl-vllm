2026-10-21 18:03:55 CST 最新结果

本轮正式4K已完成并CPU完整恢复审核通过：四轮5451.72/5615.61/5620.41/5618.02 total tok/s，TTFT5549.84/3757.04/3911.58/3837.75ms，均256成功。排第一轮后三轮均值5618.01 total/1123.60 output、TTFT3835.46ms，吞吐比5586.28高.5680%，TTFT比3780.63慢1.4503%，spread .0854395%，0抢占、0后就绪调优。用户已收到四轮逐轮成绩与最终数据。官方Math101/105，C64补充103/105。

原watcher249716已Z，controller255255/API255257消失，ixsmi68MiB空闲。原result=phase audit_failed且error KeyError('workload')，保留未改。独立experiments/tianshu_nn_silu_fx_official_recovery_20261021/recover_cpu.py只从四行CSV/未改命令推导缺失workload，CPU副本只加该字段，原runtime/Math/benchmark/C64全部审核器重新完整执行通过，零抢占和64/4K帧断言同样执行；全部旧审核源字节匹配GPU前21f9a0e，无新GPU。OFFICIAL_RECOVERY_GATE_32g.json为完整恢复证据，完整profile仅容器/workspace/benchmark-results/20261021_nn_silu_fx_official_CPU_recovery_tianshu/full_comparison.json并有SHA。最终报告及四轮CSV在tianshu_nn_silu_fx_official_results_20261021/REPORT_32g.md、ROUNDS_32g.csv、SUMMARY_32g.csv。

已新增tianshu_nn_silu_fx_release_20261021/serve.py，--output [新目录]启动同一450-source/62环境/canonical FutureKV candidate/13原argv配置；--prepare-only只CPU。已实际CPU dry-run通过，不启动服务。此候选按总吞吐记录新高，原5586.28标签保留比较，TTFT小退已如实报告。接下来提交并tag/push本次完整成果，然后开始用户要求M范围测试。

M范围BF16 QKV/O/Down三个role完整首个driverready、3852 CPU index反演和语法通过、入口实际active进程返回3证明防并行。候选源4d24695已push，此后未改：probe_bf16.py最终含W静态CPU pack/H2D耗时、42层新增bytes估算。GPU空闲且当前归档push完成后bash experiments/tianshu_m_range_stages_20261021/run_bf16.sh。root=/workspace/benchmark-results/20261021_m_range_BF16_GPU_tianshu；输出见上一STATE，对成功GPU跑verify_bf16_cpu.py --root [root]。任何运行后源/PLAN/CPU gate冻结，代码修复新目录；仅metadata错误CPU恢复，不重跑GPU。

INT8 GateUp/LM和完整MLP driver仍要完成，别缩小用户scope。当前prefill源真实NN布局、24warm_M，未覆盖prefill M保留BF16。后续比较当前unfused grouped-NN、row-major融合和grouped融合，分离quantize/activation，再按M范围加密和42层轮换/实际4K服务；新的int8_fused_grouped.py仅schedule不同、计算suffix AST相同，尚未GPU。Q2准备重复且arena不等价，已REVIEW，不运行。

以下旧状态均是历史记录，以以上最新为准。硬停仍2026-10-21 23:58北京/15:58UTC，目标继续，不自行暂停/完成。所有代码/报告/关键结果容器和GitHub，不留本地；旧user dirty两GateUp源及CPU logs不动。

2026-10-21 17:47:09 CST 最新继续点

正式4K第1轮5451.72/1090.34 tok/s、TTFT5549.84ms、240.42s，N256全成功（按预案排除）；第2轮5615.61/1123.12、TTFT3757.04ms、233.41s，全成功。已分别向用户报数。尚未新最终纪录，第3轮活跃；watcher249716/controller255255/API255257。STATE旧文starting_server已过时；root=/workspace/benchmark-results/20261021_nn_silu_fx_official4k_tianshu，benchmark_results/raw_runs_20261021_173653.csv为逐轮权威数。当前关键两轮已复制至tianshu_nn_silu_fx_official_results_20261021/ROUNDS_LIVE_32g.csv，完成后用完整四轮及comparison审核更新，不把LIVE当完成。普通Math101/105、C64补充103/105均通过。

用户要求当前正式测试完成再试M范围。M目录已有完整BF16 QKV/O/Down driver+CPU syntax/3852个索引反演通过：probe_bf16.py、matched.py（与旧天数匹配layout对照完全同bytes）、run_bf16.sh、verify_bf16_cpu.py、CPU_BF16_PREPARATION_32g.json。实际调用run_bf16.sh在watcher活跃时返回3且GPUroot未创建。主GPU入口代码准备完成，当前还没执行GPU；全部source/PLAN最终freeze后勿改已运行源，新增修复用新目录。

BF16程序三角色24M×2相位，native library mm_out、同tile NT、W-packed、A+W-packed，加单独A_pack timing，7轮轮换×20真实Graph。完整输出只在GPU在线对照FP32；独立CPU重新加载保存的真实选行列F64参考，不冒充保存所有完整输出。native library对照不代表整个当前模型（特别M2048的special BF16 kernel），筛选后仍需42层轮换与真实4K服务。GPU空闲后可bash experiments/tianshu_m_range_stages_20261021/run_bf16.sh，root=/workspace/benchmark-results/20261021_m_range_BF16_GPU_tianshu；成功后python3 .../verify_bf16_cpu.py --root [该root]。

重要新核对：tianshu_int8_gateup_prefill_grouped_20261021/kernel.py虽docstring说NT，真实W地址W+absolute*N+columns，是连续[K,N]的NN。不要把将prefill NT改NN当新假设。M目录int8_fused_grouped.py仅新增GROUP_M调度，K-index setup起完整计算AST suffix相同，GROUP1还原row-major；GROUPED_FUSION_PREPARATION记录，尚无GPU/完整driver。后续优先准备当前unfused grouped-NN vs row-major fused vs grouped-fused的M范围探针，并分别计A8量化/激活，保留同一W8 bank；LM与完整MLP driver也待做。不要因BF16首个driverready就缩小用户全部角色/重排/融合/范围的目标。

所有新增源尚无GPU收益结论。Q2旧准备已REVIEW判定重复且arena不等价，不运行。下方是较早历史状态，以上最新信息优先。

2026-10-21 17:36 北京时间最新更新：下方17:34的C64运行描述已成为历史。

C64质量完整103/105=98.095238%、105成功/0错误、103stop+2max32768，冻结C64_QUALITY_GATE_32g.json审核通过（83KB），42层融合实际replay和LM64 INT8覆盖、450数值源同源、唯一FutureKV准入差异均true。原质量controller/API/EvalScope均终态/消失。

watcher249716现phase=official4K，正式controller255255/API255257已启动，root=/workspace/benchmark-results/20261021_nn_silu_fx_official4k_tianshu，仍starting_server（17:35实查live，原命令）。继续盯四轮C64/N2564K/1K吞吐/TTFT，每完成一轮向用户报数，别因为启动几分钟或观察超时重跑。原Math8的101/105、C64补充103/105均通过。完成正式审核、记录与push后，再做用户指定的M区间方向；不运行已注明重复的Q2准备组件。

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
