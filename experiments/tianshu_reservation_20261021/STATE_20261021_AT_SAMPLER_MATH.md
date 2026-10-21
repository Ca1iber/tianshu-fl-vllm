# 当前TopP候选已开始独立Math

2026-10-21；预约至10-22 00:00，GPU硬停23:58北京。推荐仍为407源prefill-int8：正式4K5586.28 tok/s，TTFT3780.63ms，Math102/105。

唯一GPU runner PID224721：tianshu_topp_grid_current_validation_20261021/run_validation.sh。runner root=/workspace/benchmark-results/20261021_topp_grid_current_validation_runner_tianshu；Math root=/workspace/benchmark-results/20261021_topp_grid_current_math_tianshu，API224729，evalscope225196。438运行源、62环境键、原13项serve，另冻13控制器源。105题/C8/T1/p.95/max32768；>95%后自动官方4K N256四轮去首轮，逐轮向用户报成绩。不因观察超时重启、不改已GPU启动目录，不重跑失败Math刷分，不测完整16K。

同次小测已通过，见tianshu_topp_grid_current_service_20261021/REPORT_SCREEN_32g.md与SCREEN_GATE_32g.json：N64后两轮5517.679→5564.004，+0.839565%，波动.011660%/.089058%，0后续调优/抢占；双方KV40609块、native显存计费相同。仅TopP几何16K/grid2改变，非重复warp16。API独立600份完整输出检查已完成，17严格阈值false行均与现有原版逐位相同并保留。新Math/正式尚未有成绩。

代码/报告/关键结果只在容器+GitHub，禁止新本地副本。origin天数Ca1iber/tianshu-fl-vllm，分支exp/tianshu-v-int8-20261020，截至c282dd7已推。两个用户dirty gateup_silu文件不碰，FlagGems clean4cdc3b7，沐曦只读。SSH master/tmp/tianshu-20261021-control，16:27前刷新。
