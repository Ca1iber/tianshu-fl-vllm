# QKV prefill INT8，基于现行407源推荐版的独立服务候选

保留当前K8V16、LM精度、NN decode GateUp及prefill GateUp INT8，唯一新算术为42层QKV的已预热prefill M（128..4096的128倍数）；decode、未知M/布局/bias走原方法，不启用O投影INT8。选中已有QKV完整九尺寸/20配置/三阶段组件中六个正收益尺寸的配置；其他预热M向上选择最近尺寸的已验证tile，必须在真实模型启动时重新完成A量化和独立CPU INT64输出检查。父双投影O失败保持，不能称父整体通过。

新增42份静态INT8 QKV和F32列scale共220631040B，保留原BF16。全部必须在原生load_model的实际DeviceMemoryProfiler尚未退出时创建，原W/version/pointer/hash保持，与checkpoint每层q/k/v拼接的独立CPU完整hash对照。候选加载后验证实际model_memory_usage覆盖原parameter+原1326938112B packed+新增QKV，禁止覆盖预算/手工改KV块数。

先CPU完整42层checkpoint/量化bank manifest、路由/预热集合/原13参数与62env绑定。GPU启动前冻结源码/selection/CPU gate；失败修复新目录。服务对照为4K输入/1K输出、C64、N64、每臂三轮排首轮，沿用现行407源全部来源/权重/真实C8与C64 Graph/usage/seed-canary/FutureKV/零抢占检查，逐轮报告。对照与候选都保持现行prefill GateUp candidate；本候选所有QKV decode捕获必须明确delegate原方法。

只有增益>.5%且超过两臂极差、各自spread<.5%、TTFT退化<=1%、排首轮autotune=0，才继续独立新105题Math，必须>95%，禁止失败后重跑刷分。若通过且小测收益稳定，再正式N256/C64四轮排首轮确认；此前推荐5586.28不变。无新完整16K。GPU23:58北京硬停，先检查空闲；原件日志/PT仅容器，代码/报告/关键JSONCSV GitHub，无本地副本，不动沐曦或用户两份dirty文件。
