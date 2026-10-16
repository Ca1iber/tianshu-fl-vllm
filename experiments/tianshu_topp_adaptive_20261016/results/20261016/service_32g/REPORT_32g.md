# 天数TopP初始阈值服务对照：不采用

以已经通过正式4K和Math的GEO+M2048版本为对照，仅将纯TopP初始outlier sigma改为max(original,std-2.5)。原始serve命令、75点预热、KV、调度及RNG保持一致，65份冻结源码、39个实际环境变量均复核。

|模式|第1轮（排除）|第2轮|第3轮|后两轮均值tok/s|
|---|---:|---:|---:|---:|
|GEO当前最佳|3854.2220|4031.2909|4038.0486|4034.6697|
|初始阈值候选|3390.2037|4036.9442|4043.2029|4040.0736|

增益+0.133934%，未达到2%复验门槛，不采用，不启动新Math或正式benchmark。每轮4K/1K、C64、N64，轮间使用不同输入，六轮384请求全部成功，所有完整文本/usage/seed/timing下载后复算。四个seed777 canary文本一致，16份metrics无抢占，六轮无新增打印的调优记录。此处是小规模筛选，非正式N256成绩。

原始目录：/workspace/benchmark-results/20261016_topp_adaptive_service_tianshu。归档SHA见ARCHIVE_MANIFEST；复核命令：`python3 experiments/tianshu_topp_adaptive_20261016/verify_service_downloaded.py --root <下载目录>`。正式最佳4093.81 tok/s、Math102/105保持。
