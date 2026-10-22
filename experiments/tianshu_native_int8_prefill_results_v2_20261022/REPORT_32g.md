# 原生INT8 GateUp：同源完整服务4K短测

两臂保留正式5695.95版本的原471源码；仅M512与4096的原生INT8实现不同。4K/1K，C64/N64，各三轮排首轮；不能作为正式N256成绩。完整继承运行时审核并独立重算384个响应的tokens、TTFT、吞吐与输入种子，实际kernel binary与完整42bank组件严格绑定。

|模式|总tok/s|输出tok/s|TTFT ms|后两轮差幅|
|---|---:|---:|---:|---:|
|control|5600.66|1120.13|10098.33|0.7140%|
|candidate|5721.43|1144.29|9526.74|0.0922%|

候选吞吐变化 +2.1563%；TTFT变化 -5.6603%。性能筛选通过：True。正式最好成绩继续引用已通过官方Math及C64质量的5695.95，候选必须独立通过新质量评测及正式4K。

所有请求成功，就绪后无autotune；完整旧KV/attention/scheduler/权重/LM检查保留，原M64已采用融合两臂相同。动态[1,4096]图在两臂均显式改写为同一个opaque op，control委托既有路径，candidate仅在M512与4096使用原生INT8；不能声称所有动态图不变。源文件与原始结果SHA记录在SCREEN_GATE；大体积响应和原始日志只留容器。

## 原生实现与显存预算

使用已有同一份 NN W8 权重和相同 A8 量化；候选只在 M512、4096 用 torch._int_mm 产生 INT32 中间结果，再按原 BF16 舍入和 SiLU 计算。没有修改原始模型、增加静态打包权重或改变 M8/M64、LM 和 KV 计算路径。两臂同源877份文件，原正式471份精确子集；原13项 serve argv 和62项环境保持一致，额外 TRITON_CACHE_DIR 明示，实际共观测63项环境。PYTHONPATH只选择control/candidate模式。

两臂都在 DeviceMemoryProfiler 中预热原生库。candidate 在原生 memory_profiling_fl 中真实运行全42层M4096，临时INT32工作区192MiB计入原生峰值和KV预算，没有手调KV大小或旁路持久工作区。旧23尺寸完整binary、新kernel binary、RNG、全部42层真实路由、原M64与attention/KV/scheduler/采样/权重/LM检查保留并独立复核。

先前单轮smoke因继承三轮请求数>=194断言而保持audit_failed；独立CPU核验仅确认其真实74个响应和接入范围。本次两臂各192个计时响应，另有canary/smoke；完整三轮审计原样通过。smoke的CPU补充核验没有替代本次完整吞吐审计。

本轮性能筛选通过：吞吐+2.1563%，TTFT-5.6603%，candidate后两轮差幅0.09224%。已转入全新官方C8 Math105，后续真实C64 Math105必须>=100/105并覆盖新M4096全42层，再跑官方4K N256四轮排首轮。此处5721.43是N64短测，不是新正式成绩。
