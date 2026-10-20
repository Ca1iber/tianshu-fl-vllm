# 当前状态：grouped-M prefill 组件运行中

预约至北京时间2026-10-22 00:00，GPU截止2026-10-21T15:58:00+00:00（23:58北京时间）。

最新已完成：GateUp NN模型小测381源码/62环境，384成功，5421.869811→5429.587101 total tok/s（+0.142336%），小于波动，未采用、未新Math/正式4K/16K。两组KV40723/40658，43-bank payload相同；完整报告在tianshu_int8_nn_gateup_service_20261021/REPORT_SCREEN_32g.md。

随后row-major prefill真INT8组件6种M×6分块，36组合/108阶段全部通过，独立CPU重读checkpoint验证通过；组件最佳M128 +37.36%、M2048 +11.35%、M4096 +13.03%，对照是当前实际BF16 linear_contiguous，A量化计入，静态W转置排除。量化相对L2 1.1726%~1.1745%，不是模型准确率。代码与关键结果在tianshu_int8_gateup_prefill_results_20261021。

唯一GPU任务：PID159457，experiments/tianshu_int8_gateup_prefill_grouped_20261021/run_component.sh，8种M×12配置（同6tile的group1/group8），包含M129行尾与M1152末组不满8。raw /workspace/benchmark-results/20261021_int8_gateup_prefill_grouped_GPU_tianshu，runner同前缀_runner_tianshu，queue同前缀_queue_tianshu.log。当前M128完成，M129进行中；必须查真实phase/result/进程，等待GPU_complete后用独立结果目录verify_gpu.py raw路径离线CPU复核，再标complete。观察超时不等于失败，不启动重复任务。

下一步：先grouped终局/CPU验证/报告；有收益则按PREFILL_MODEL_INTEGRATION_PLAN_20261021.md接入模型，对prefill路由审计作明确适配，不伪造旧BF16请求命中。Math/正式4K仅收益明确后。当前正式默认仍5482.79 / Math101/105；高吞吐备选5509.23 / Math102/105，较高TTFT非默认；16K旧配置3885.91。没有本地源码或结果副本，原件仅容器，代码报告关键结果推GitHub。
