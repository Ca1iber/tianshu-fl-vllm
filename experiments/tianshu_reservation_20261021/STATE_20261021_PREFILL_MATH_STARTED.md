# Prefill INT8小测完成，启动全新Math→正式4K

预约至2026-10-22 00:00北京时间，GPU23:58硬停止。最新小测407来源/62环境/原13参数，计分5425.622762→5520.640252 total tok/s（+1.751273%），TTFT10896.638383→10369.043518ms（-4.841813%），两组波动0.1484%/0.1402%，384成功，计分轮无现场调优、零抢占、完整审计通过。两组NN纯decode相同，候选仅24个对齐M的prefill/混合矩阵使用INT8。

experiments/tianshu_prefill_int8_validation_20261021/run_validation.sh：先全新Math105，严格>95%后自动执行原始正式4K N256×4轮排首轮，无16K。源码准备门和完整Math/official CPU预检查已通过，两阶段精确复用407运行来源、62环境、13参数、相同额外CPU controllers；不更改运行profile。Math按C8真实路径、官方按C64/4K，均验证42层新prefill，不能套用C64请求要求到Math。

唯一后续GPU队列 /workspace/benchmark-results/20261021_prefill_int8_validation_queue_tianshu.log/.pid，runner同前缀_runner_tianshu。Math raw /workspace/benchmark-results/20261021_prefill_int8_math_tianshu，official raw /workspace/benchmark-results/20261021_prefill_int8_official4k_tianshu。检查实际进程/阶段，不能把poll超时当停止，也不能重复启动或为准确率失败重跑求过线。质量或审计失败都先保留终局证据再独立修复/新版本，不改已冻结控制器。

当前没有新的Math/正式成绩。正式默认仍5482.79、Math101/105；高吞吐备选5509.23、Math102/105、较高TTFT非默认。用户两个dirty文件不动，所有原件只容器，代码报告关键结果推GitHub。
