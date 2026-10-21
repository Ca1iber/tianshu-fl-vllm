2026-10-21 17:19 北京时间

# 继续点

官方Math8已真实终态：runner247467=Z，API247478/EvalScope247928已消失。raw=/workspace/benchmark-results/20261021_nn_silu_fx_math8_tianshu，result.complete，101/105=96.190476%，0请求错误，104正常stop+1达到32768。冻结framework_profile的audit_runtime及audit_math已在CPU完整通过，accuracy_audit.json保存完整4.5MB逐层记录；Git关键摘要86KB在tianshu_nn_silu_fx_math8_results_20261021/MATH_GATE_32g.json，9aedfd6。没有复用旧102/105，没有重复GPU跑分。

C64补充/正式4K串行watcher249716仍活跃，phase=C64_math。补充controller250826、API250828，截至此刻starting_server（启动实际要几分钟，不因观察超时重启）。root=/workspace/benchmark-results/20261021_nn_silu_fx_c64_quality_tianshu。入口2706c9e已freeze/push；全105题/max32768，62环境唯一FutureKV candidate→control，450数值源保持，原生allocator保持，必须实际42层M64融合replay>=100和LM64 INT8>=10，准确率严格>95%，成功后才自动正式N2564K。官方Math8本身不能覆盖M64融合。

补充gate=experiments/tianshu_nn_silu_fx_c64_quality_20261021/C64_QUALITY_GATE_32g.json，正式root=/workspace/benchmark-results/20261021_nn_silu_fx_official4k_tianshu，C64 N256 4K/1K四轮排首轮。若质量失败，不重复抽分；若仅审核metadata错误，保留原failed并CPU新目录恢复，不重跑GPU补metadata，不降低准确率或捏造覆盖。每完整benchmark轮完成向用户报数。所有已启动代码/CPU gate/PLAN保持冻结。

新独立组件已准备并push 40efaa4：experiments/tianshu_k8v16_q2_direct_20261021。source只把已合格f1df pair-rows reader的3D wrapper max_q>1改>2，所有device函数AST不变，generated1475fff6...。按真实query tokens分配128token缓冲，894组CPU Q1/Q2映射/26925token通过。probe比较T16W1/T32W2/T64W4，uniform Q2用真实BoundedDensePrefill+hybrid FlagGems BF16对照，混合只数值验证，不声称简化baseline服务收益。全输出FP32因果参考+独立CPU抽head、改变QKV/页表/长度两phase、9交替×20真实图重放；不运行模型，不算新吞吐纪录。

组件尚未GPU执行。run_component.sh已实际调用验证在watcher活跃时返回3且未创建GPUroot。必须等watcher249716真终态、ixsmi空闲后才手动启动：bash experiments/tianshu_k8v16_q2_direct_20261021/run_component.sh。root=/workspace/benchmark-results/20261021_k8v16_q2_direct_GPU_tianshu。成功后CPU核对：python3 experiments/tianshu_k8v16_q2_direct_20261021/verify_cpu.py --root [该root]，然后写报告和push关键结果；任何失败不改原冻结代码重跑，修复新目录。组件即使好也需真实投机None guards/混合图/M128 GEMM/质量验证，不能直接采用。

当前450候选三臂小测：原始407参考5508.9690，静态64对照5529.4693，融合5554.9744 total tok/s；整体+0.83510%，融合相对静态+.46126%；576请求全成功，零后就绪调优。未完成新正式N256，当前正式最好仍5586.28 total /1117.25 output，TTFT3780.63ms，407源Math102/105。

预约到2026-10-22 00:00，北京时间10-21 23:58硬停/15:58 UTC。当前17:19，仍有6h39可推进，不自行暂停/完成目标。SSH master=/tmp/tianshu-20261021-afternoon-control（10h）；旧master已过期。代码/报告/关键结果仅天数容器与GitHub，原始大日志/张量容器，不留本地参考。GitHub分支exp/tianshu-v-int8-20261020，push仍需主机stdio FIFO桥。两旧GateUp用户dirty文件及旧CPU log不动，不改沐曦仓库。
