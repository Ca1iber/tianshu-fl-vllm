# LM补齐M32–64新完整正确率门槛

固定候选在原r3（903源精确保留）上仅增加LM32..64补齐64及FP32输出融合；M64与原A8路径逐位一致，32..63是新量化；M<=8保留原BF16，原QKV128量化保留，未加入被淘汰的QKV64。先新完整105题C64（FutureKV control），通过>95%后新完整105题C8（FutureKV candidate），两者模型源一致。原官方Math-500 Level3、temperature1/top_p.95/max32768/timeout3600/ignore-errors；所有105题完整计入，长度上限/错误不能删除，不原样复测挑分。

原13条serve argv、完整63环境变量，所有源/控制器快照/原checkpoint SHA。实际GPU源与新LM CUBIN、原NT pack/scales、实际loader DeviceMemoryProfiler零64预热与真实memory_profiling显式zero64探针、无额外权重/全局workspace均核验。显式探针不冒称真实请求；实际LM运行M范围/次数单独记录。旧dense_audit仅LM64旧kernel调用计数由新LM实际整数路由计数及组件二进制证明替换，旧parent组合与GU64图100重放、LM8 BF16、完整43bank、其他attention/KV/scheduler/sampling/环境/RNG/零抢占等核验全部保留，返回状态明确旧LM64执行不再声称，不能伪造旧binding。

该新控制器全部代码与计划在首次GPU前冻结。CPU预检两组源/控制器一致、原argv及全环境、审计替换目标真实存在。2GiB磁盘启动最低门槛：复用原r3编译图，新增部分只有LM外部算子和少量源快照，开机前测实际剩余；所有旧严格384缓存检查保持不变，不清除活跃cache。GPU串行且启动前空闲；绝对硬停止2026-10-22T15:58Z，结束核验空闲。只在独立双质量合格后短4K，>0.5%收益/热轮极差<1%/TTFT<=1.01才原N256四轮；不跑16K，不提前采用。

首个服务GPU前将组件已验证的两个Triton cache group复制到原指定qualified_model_cache的新唯一目录，只改新group JSON的child_paths为新绝对目录，不改CUBIN或旧384文件。LM_COMPONENT_CACHE.sha256逐文件冻结两个新group，实际启动前同时检查新group清单与原384严格缓存。纯CPU控制器retry1重检包含该清单；旧预检Gate保留独立文件。服务925模型输入不变，质量控制器增加缓存清单；未改任何已在GPU运行的模型源。
