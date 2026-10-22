# 原生INT8预填充真实服务 V1

固定全42bank holdout的NN/ROWS1/BLOCK1024/W8，仅M512/4096。current control=现用5695.95全配置，candidate替换这两M完整GateUp；其他M/所有decode/K8V16/cache/sampling/原BF16权重及所有43个W8 bank不变。新FX pass仅在parent的dynamic[1,4096]把旧range customop target改同签名新opaque op，fake保持BF16[M,6144]。双方都用新target，control委托旧op；禁止伪造旧kernel调用。

候选每调用分配临时q/scales/INT32/out，同一stream顺序量化→torch._int_mm(packed.T)→原SiLU epilogue。无新增物理W layout，无全局tensor side-channel/scratch。Native persistent库/模块在42nd NN bank处理、原DeviceMemoryProfiler仍活动时两组同样零输入warm；warm与RNG/签名核验独立记录。真正profile_run阶段candidate必须实际执行M4096新路径全部42bank，且Worker原生memory_profiling上下文活动、手动KV bytes未开启，峰值由原生预算计入；不能等KV分配/capture后才开启。workspace201326592 bytes不得隐藏为不计费常驻缓冲。

保留原471数值源、62个既有选定环境与13个serve参数。另显式审核TRITON_CACHE_DIR，实际选定环境合计63项；旧62项观察文件保持其原范围，额外完整63项记录独立核验。不把62项称为整个进程环境。24点qualified cache及group实际来源逐文件冻结，full binary匹配保留。

新原生route与旧range实际route共同覆盖所有42bank，candidate M4096运行期须新route全部42、control须旧route全部42，不能借load warm/profile/capture调用当请求命中。继承attention/KV/scheduler/LM/M8与M64图重放审计，只替换真实prefill覆盖适配；cache/新profile单独审核。CPU fake/FX/边界先检查，冻结后启动；首次smoke只64个4K/128及C8 canary，非正式速度/质量。成功再两臂4K/1K/C64/N64三轮排首轮、小样本稳定>0.5%后fresh C8+C64 Math≥100/105，最后正式N256四轮。不要常测16K。

所有source先冻结，失败源码/raw保留，修复新版本。GPU每天06:00–24:00，今日23:58硬停，startup30min、整体按截止控制。仅容器与天数GitHub，代码报告关键数据推送，raw仅容器；原模型/沐曦与用户脏文件不动。

## V2 记录命名隔离

V1 的 GPU 启动失败与数值计算无关：native_prefill_records 已被继承的 attention 探针占用。V2 所有新增 INT8 记录改为 native_int8_prefill_records；继承的 attention 探针路径、62 项环境和 471 份源码保持不变。原失败源码和原始结果保留在 V1。
