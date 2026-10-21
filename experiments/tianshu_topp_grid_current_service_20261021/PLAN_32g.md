# 当前推荐版 + TopP16K/grid2 服务对照

两臂都继承407-source/62-env推荐prefill INT8/NN/LM/K8V16/FutureKV，原13serve参数不变；只有采样几何在支持的B8..64改变，GEO源09156e66...、p/温度/RNG和Sampler/alias保持。API已有600完整输出/28.8亿FP32重建、22056行FP64语义、33RNG/8生命周期/8fallback门槛；17个strict false仍false但等基线。新service自己的kernel_route仅在GPU launch之后缓存源码/binary审计哈希，CPU AST证明整个GPU启动前缀与已验证API完全相同；不将候选每步文件读取算成优化成本。

两臂在真实memory_profiling_fl上下文进入后、yield给原profile_run前，预热原GEO32K和候选16K对全部B8..64，共114次；每次恢复未过滤logits，全局RNG不变。64行FP32共享scratch33423360B、native sampler cache、临时和JIT模块在原生计量范围，无手工resetpeak。parent capture完整返回之后才启用候选，原模型Graph不改；ready marker之后才标记runtime，Python调用计数不冒充Graph回放。

原C1 seed canary保持，另有未计时8×128/16联合请求保证实际C8采样路径命中。4K/1K C64/N64每臂3轮排首轮，输入种子跨臂配对，逐轮报告。新增显存真实native计算KV预算，source/实际环境/43bank/缓存/Graph/FutureKV/零抢占全部父audit用candidate模式保留，新audit另标新几何模式。稳定明确才新105Math严格>95与正式N256，不常规16K。

API的数值源已冻结，服务启动后本目录py/sh/PLAN/CPU_GATE/selected_plan也冻结；真修复新目录，元数据独立CPU恢复不重复GPU。串行GPU、每臂最多1h、23:58北京硬停。代码报告关键结果GitHub，raw仅容器；两处用户dirty及Ops不动，无本地副本、无子agent。
