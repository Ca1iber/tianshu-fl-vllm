2026-10-21T17:02:26+08:00

# 活动验证与继续点

官方Math8 runner PID247467，API PID247478，EvalScope PID247928。root=/workspace/benchmark-results/20261021_nn_silu_fx_math8_tianshu；尚在evaluating，正确率未出。450数值源/62环境/13原参数与已屏测候选完全相同。必须先ps确认同一进程和result/state，不因观察超时重跑。

补充C64质量及正式4K的串行等待runner PID249716，phase=/workspace/benchmark-results/20261021_nn_silu_fx_c64_quality_runner_tianshu/phase.txt。它现在仅CPU等待官方Math8确实终态；只有完整105题且严格>95%、GPU空闲才自动启动C64质量，C64完整通过后再自动启动正式4K。其入口2706c9e已push，所有py/sh/CPU gate/计划已冻结，不修改；修复新目录。

C64质量仍全部105题/max_tokens32768，仅FutureKV candidate→control，原生allocator保留，同450模型源、61其余环境按root归一化保持。要验证42层M64真实replay>=100、LM64整数调用>=10、真实大量准入、原生allocator和全权重/动态/M8/prefill保留；必须严格>95%。原始Math8不覆盖M64，二者分开，不复用旧102/105，不截短答案或筛问题。直接审核依赖已在辅助CPU v2额外冻结；Math8后补依赖文本只证明与GPU前Git21f9a0e相同，不冒充API启动时快照。

辅助root=/workspace/benchmark-results/20261021_nn_silu_fx_c64_quality_tianshu；gate=experiments/tianshu_nn_silu_fx_c64_quality_20261021/C64_QUALITY_GATE_32g.json（尚不存在，等待实际执行）。若同一GPU结果只因审核metadata失败，保持原始failed，CPU新目录恢复，不重跑GPU补metadata；不得放宽准确率或捏造覆盖。

两项通过后自动正式root=/workspace/benchmark-results/20261021_nn_silu_fx_official4k_tianshu，原benchmark 4K/1K C64 N256四轮排首轮。官方审核强制读辅助gate、验证450源相同和唯一准入差异，同时新鲜官方Math8与正式source/env/controller绑定。不能在辅助未通过时手工绕过门槛。关注每完成一轮向用户报吞吐和TTFT。

已完成三臂小测：原始407参考5508.9690，对照静态64=5529.4693，融合=5554.9744 total tok/s；整体+0.83510%，融合相对静态+0.46126%，静态自身+0.37212%。后两轮spread分别.15652/.09642/.07679%，576请求与全审核通过，0后就绪调优。候选过预设性能筛选，尚未采用，正式最好仍5586.28/1117.25，TTFT3780.63ms，原407版Math102/105。关键SCREEN_GATE/ROUNDS/CSV/REPORT在fx_screen目录，a07e216已push。

最新SSH master本地 /tmp/tianshu-20261021-afternoon-control（10h，不用已过期旧master）。GPU截止2026-10-21 23:58北京时间/15:58 UTC，预约到10-22 00:00。目标继续有效；验证结束后继续找收益，不自行暂停或完成。

后续CPU独立方向可读投机历史REVIEW：GPU ngram旧接受29.11%但全eager；保持真实请求数的248个mixed键早已实现，84.73%重放且+6.51%，仍慢于非投机。不能重复宣称mixed图问题未处理。真正可核对的新缺口是当前压缩K8/VBF16 attention是否能直接处理Q2小查询、避免spec验证走BF16 materialization；先读实际3D路径/旧报告和None guards，再设计组件，尚未实现或跑GPU。K6/7压缩、现代长输出采样融合也仅为假设，未实现，不误报。

用户的两个dirty GateUp旧文件及其他未跟踪CPU日志保持；不改沐曦仓库。代码/报告/关键结果只容器与GitHub，不留本地副本，常规操作直接shell/git。GitHub分支exp/tianshu-v-int8-20261020，源已push2706c9e，接下来只需push本STATE报告。
