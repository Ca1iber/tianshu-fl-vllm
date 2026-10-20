# 2026-10-21：大 M INT8 模型配对准备完成

许可至2026-10-22 00:00北京时间，GPU统一截止23:58。grouped prefill组件96配置/288阶段和独立CPU复核全过：M4096 BF16 2.364668→INT8 2.012801ms（组件换算吞吐+17.48%），M2048 +16.76%，M129全负，保留BF16。

当前启动 experiments/tianshu_prefill_int8_service_20261021/run_pair.sh：407源码/62环境/原13参数/4096预算/64序列/async，control与candidate均继承相同NN纯decode布局；候选仅已预热、128对齐contiguous大M矩阵（含混合行）用INT8。实际中间M仅有启动数值证明，最终收益看模型。启动预热默认约24点；真实数量以worker记录为准。未计分种子canary使用唯一cache_salt强制两次完整重算，计时请求不加seed/盐。

队列 /workspace/benchmark-results/20261021_prefill_int8_service_queue_tianshu.log/.pid；runner同前缀_pair_runner_tianshu；两组raw同前缀_{control,candidate}_tianshu。观察超时不等于终局，检查实际PID、phase、result和日志；不得并行启动第二个GPU任务或重跑求运气。

已完成NN decode模型小测仅+0.142336%，未采用；prefill模型还没有成绩/新Math/正式benchmark。正式默认仍5482.79、Math101/105；高吞吐备选5509.23、Math102/105、较高TTFT非默认；16K旧配置3885.91。若新小测明确提升，必须新Math105题严格>95%再正式4K。两个用户dirty gateup_silu文件不改不提交，所有源码结果不留本地。
