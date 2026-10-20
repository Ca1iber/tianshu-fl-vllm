# 新默认5586.28已复核，继续Down→KV组件

预约至2026-10-22 00:00北京时间；各GPU任务23:58 UTC15:58硬停止。新默认标签tianshu-original-command-4k-prefill-int8-20261021，正式5586.28 total/1117.25 output、TTFT3780.63ms，Math102/105（97.142857%）。相对前默认+1.88754%、TTFT-5.84275%，相对公开+175.45624%。1024成功、零错/抢占、post-ready tune0，吞吐极差0.13927%，全独立复核采用true。MathKV40731、正式40723差8块已公开。两者同407运行来源、12主controller、62env、原13argv，额外quality/official两个CPU复核器各记SHA。正式原件20261021_prefill_int8_official4k_tianshu，Math同前缀_math_tianshu，validation父runner已complete，不重复跑。

新release Bash原命令/26份生成依赖，最终dry与原dry字节相同；README和TIANSHU_BEST已更新，report/gates/原CSV/summary保持原值，原comparison容器保留，GitHub仅profile以外比较摘要及原件SHA。新配置没有16K，3885.91属于旧FutureKV；不安排例行全16K。

唯一GPU接续队列PID文件 /workspace/benchmark-results/20261021_after_prefill_components_queue_tianshu.pid；runner /workspace/benchmark-results/20261021_after_prefill_components_runner_tianshu/phase.txt。已在正式结束后进入down_component。Down目录tianshu_int8_down_grouped_20261021，9M×20 tiles/group（包括BM16/32），真实DownW，基线真实mm保留旧M2048专用tile，另测vendor linear。编译证据从当前Inductor mm.out到同一FlagGems_launch_mm已保存。raw /workspace/benchmark-results/20261021_int8_down_grouped_GPU_tianshu；CPU结果目录tianshu_int8_down_grouped_results_20261021（verify_gpu.py会生成GPU_GATE/CSV/rounds）。仅在GPU及独立CPU复核都通过后写模型实验，不把组件收益直接当模型。

然后自动KV fixedk_pairv组件（tianshu_k8vi8_fixedk_pairv_20261021），四路相同writer/页布局，nativeV8、恒1 K-scale消除、再加成对Vload，对照K8V16。CPU已8原生layout/65536码值通过；新reader相对V2三处可逆变更。raw /workspace/benchmark-results/20261021_k8vi8_fixedk_pairv_GPU_tianshu；run后verify_gpu.py及ROUNDS_32g.json。两组件source已冻结，不在GPU运行时改.py/.sh或CPU_GATE，失败保留后另建恢复版本。CPU准备必须CUDA_VISIBLE_DEVICES=-1（原可见导入会初始化上下文的失败已记录），不可与正式计分GPU计算并行。

待办：监控两个组件，完成各阶段及时报成绩；复核后写报告并push关键结果，评估Down的M8/64及largeM、BF16 linear有无真实收益。KV noKscale+PairV若仍负不做模型。若Down正收益，单独模型N64三轮排首轮、同一profile对照；显著后新Math105>95和正式N256四轮，再发布。绝不为正确率失败重跑求过线。MetaX只读参考，用户两个gateup_silu dirty文件不碰。所有完整原件仅容器，代码报告关键评测结果GitHub；本地不建副本。
