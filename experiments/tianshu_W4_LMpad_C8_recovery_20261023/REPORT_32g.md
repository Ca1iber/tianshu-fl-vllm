# C8 Math-500 CPU 复核

2026-10-23，LM M1..64 W4（M1/2直算，其余补齐32/64）与Down64 W4组合，模型冻结944份源码。原始GPU控制器实际97份，未改动任何一个。C8一次完整评测105题，100正确，95.238095%；105请求均成功，错误索引18、45、83、87、101；104次stop、1次触及生成上限，全部纳入分母。

原始运行结束为audit_failed / AssertionError((8, 0))。原因是继承的C8检查要求旧NT8预热捕获，当前模型的最终图实际使用NN8。原始NN记录显示42层真实捕获，绑定的Graph8 capture_calls=1、replay_calls=39138。CPU复核只替换这项旧捕获要求，检查真实NN配置、冻结源码、42层前缀和图回放关联，同时保留旧NT源码与配置检查以及全部后续运行检查。

CPU复核通过，无GPU重跑、无预测筛选、无评分改动。原始GPU结果仍为audit_failed，单独复核目录标记CPU_recovery_complete、original_experiment_passed=false、GPU_rerun=false。当前模型权重文件SHA保持14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c。

初次CPU复核因从父版本沿用95控制器的计数而提前失败，保留INITIAL_EXECUTION及日志；核对原始快照确认为97后更新CPU计数并重新冻结。此失败未触及GPU或原始记录。

C64同944源码版本已启动；未通过双批次正确率和预先规定的短测门槛前，不晋升到完整4K测试，不替换已采用r3。当前无新的端到端性能成绩。

原始证据：/workspace/benchmark-results/20261023_W4_LMpad_v2_math8_tianshu。
CPU复核：/workspace/benchmark-results/20261023_W4_LMpad_C8_CPU_recovery_tianshu。
