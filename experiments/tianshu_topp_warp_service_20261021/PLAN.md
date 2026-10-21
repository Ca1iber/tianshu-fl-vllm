# 当前推荐模型 + TopP 16 warp 小规模配对

控制和候选都继承407-source prefll INT8推荐版（5586.28正式/Math102）。原始13 serve参数、62显式配置、所有NN/INT8/43bank/K8V16/Graph/调度保持一致。新源数由CPU实际prepare，两臂同闭包；唯一请求路径变化是原始TopP JIT的num_warps4→16，只在CUDA FP32/130560词表/连续行/B8..64/TopK关闭/TopP开启。其余委派native，算法/temperature/top_p/RNG未改。父profile.audit_runtime用candidate模式保留全部原检查，新审计另查sampler真实ready后命中/源码/compiled binary/计费。

连续组件已有180全输出/846028800值/1272CPU核验，W16过滤耗时少28-35%，但与native不是全部逐位相等。旧bitwise gate false原样保留。p≈.95有效行最大截断TV .00186977、无rank inversion；原生自身也与FP64 ideal有差异。本新模型明确是允许FP32归约顺序变化的探索候选，不宣称数值完全等价。按用户授权以新Math105严格>95%和真实模型收益判断，失败不抽样重跑；组件/旧Math不当新模型成绩。

两臂在真实原生memory_profiling_fl context内部、yield给原profile_run之前预热W4和W16的B8..64共114次。预热数据确定性、全局CPU/GPU RNG必须不变，无手工reset peak；原生Peak读取发生随后，审计要求最终activation峰值覆盖warm峰值。新增native buffer/CDF/cache/临时/模块均在计量范围，warm临时张量释放，原cache继续复用。完整父capture_model返回之后才激活候选；ready后实际request记录与post-capture暖身分开，不伪称目标Graph捕获sampler。

每臂原始命令4K/1K C64/N64三轮，排首轮，不常规16K；input种子每轮变化、跨臂配对，保留384完整响应和指标。两次seeded B1 canary走原fallback，要求自身/跨臂相同；八请求真实INT8 smoke、全部父源码/权重/native/FutureKV/Graph/抢占审计保留。测量后两轮autotune/容量/TTFT/波动单独披露，局部filter提升不换算模型吞吐。收益稳定明确才独立新Math与正式N256。

GPU启动后本目录全部py/sh/PLAN/selected_plan/CPU_GATE冻结，真正代码修复新目录；纯元数据归档恢复独立CPU目录，不重跑GPU。代码/原件只在容器和GitHub，Git只代码报告关键JSON，不碰用户两处dirty/沐曦仓库，不启动子agent。唯一串行GPU，臂最多1h，23:58北京硬截止；每完成轮报告。
