# 2026-10-21 09:46 北京时间：多查询组件完成，投机语义核查中

预约到 2026-10-22 00:00，北京时间 23:58 硬停止 GPU。当前没有 GPU 任务，ixsmi 空闲 68MiB。SSH master /tmp/tianshu-20261021-control，04:27 建立，16:27 前刷新。

推荐版保持 tag tianshu-original-command-4k-prefill-int8-20261021，完整 4K 5586.28 total /1117.25 output tok/s，TTFT 3780.63ms，Math 102/105。预算 8192 正式 5608.40（+0.396%），TTFT 4406.24 变差，不采用；静态编译 +0.254%、KV 边界 +0.0543% 不采用。此次多查询只为投机准备，普通 4K 未观测到 Q≤8 命中。

multiquery_paths_GPU_tianshu 完成 26×5、两输入 phase、7轮×20replay；25源冻结闭包不改。原 verifier 索取不存在的逐 phase native arena guard 字段，独立 CPU 恢复器位于 tianshu_multiquery_paths_recovery_20261021；原结果/源/CPU gate 不改，未重跑 GPU。复核 260 完整输出/1096理想向量/117988352codec值/50原生BF16-P短K全输出/910计时。报告 tianshu_multiquery_paths_results_20261021/REPORT.md。C64Q3K4096 当前路径23.971686ms→原Split-KV .892061ms；新T16不胜。C64Q4 新T16 1.169128ms 比原3D 1.727730少32.3%。不能当模型成绩。逐phase输出/scratch/KV/RNG检查，native arena/IDs仅逐case全部计时后检查，报告明确范围。

当前新方向：先核查旧 DSpark 接受率低是否符合语义。旧 prefix2_standard_4k1k 第三轮 drafts248052-165862=82190，drafttokens496104-331724=164380，accepted145174-96320=48854，每验证平均接受 .5944 草稿token；位置0 30349、位置1 18505。旧吞吐第三轮1948.91398，旧NoSpec2731，不可与当前5586相混。目标与草稿配置 logit_scale 默认1，无缩放差异。官方当前vLLM DSpark也继承DFlashQwen3Model，不能仅因本地DFlash无projector_type分支就认定移植错。需继续比FC/hidden_norm、辅助层索引、anchor/context position、Markov、自回归采样；模型卡自然prompt7步长度与本地随机输入2步不直接可比。安装版.24无原生DSpark，不能直接改method。

当前407默认有SpecNone和42因果层假设，不能叠加旧DSpark。真正新集成须47层42因果+5非因果，新增scratch/权重/arena均原生DeviceMemoryProfiler计费，标准正FP32 rejection，所有实际b1..64草稿7b/context3b/head2b/Markov启动预热，新GPU目录冻结后不改源。先语义诊断再模型对照。不要重复负收益微调或完整16K。

仅容器和GitHub保留代码/报告/关键结果；不在本地复制。保留用户两处 gateup_silu dirty。Ops4cdc3b7不改。不启动子agent。直接shell/git，不用Python包装文件操作。每完成benchmark一轮报成绩，持续更新；目标仍active。
