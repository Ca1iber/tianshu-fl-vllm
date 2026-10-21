# 逐M GateUp融合：新官方Math8、真实C64补充质量及正式4K

只有同源4K短测完整通过、后两轮差幅每臂<1%、候选吞吐>.5%才启动GPU。先全105题官方Level3，eval batch8 / temperature1 / top_p.95 / max_tokens32768，严格>95%（至少100/105）；不得重抽失败质量。然后同471源码、同原13参数serve及62环境，唯一未来KV准入开关candidate→control（原生allocator）以真实C64补充105题，要求42原M64融合bank图真实replay覆盖与LM INT8实跑，严格>95%；不能以启动capture/虚假dummy代替实际C64请求。正式4K仍用原candidate准入开关。

运行时审计分别完整继承旧Math8的K8V16/attention/allocator/seq-limit/43bank权重与LM BF16检查、真实C64的原生准入及旧42bank融合图replay/LM INT8检查；原M64 pass动态不变断言只针对旧pass。仅原prefill实际执行条件适配到新range fused记录，另完整审核新86图pass、23M启动独立CPU真W检查、实际二进制与3024场景组件严格一致、42bank实际runtime fused路由、旧450源为exact subset、无新增packed payload，明确动态[1,4096]已经改写。模型source map471必须与短测逐字一致，控制器源码另行冻结；Math8、C64、正式均用同一控制器冻结map，C64只改准入环境不改source。

质量都通过才跑未改原benchmark脚本N256四轮，排首轮，逐轮报告，0preemption/0就绪后autotune/所有1024请求成功。正式controller显式写入workload避免上次metadata缺失；若失败保留原始目录，不改已GPU运行源码，不为补metadata重跑GPU。CPU先真实prepare两种环境，不初始化CUDA。控制器入口/PLAN/CPUgate首次GPU前SHA冻结。GPU串行，不与仍在运行的pair共用卡，23:58北京硬停，时间不够不启动下一阶段。原正式5618.01及tag继续保留，只有新验证全部完成才新增版本，不引用组件收益作为模型成绩。
