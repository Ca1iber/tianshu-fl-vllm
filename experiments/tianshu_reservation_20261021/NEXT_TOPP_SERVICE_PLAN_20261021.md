# Top-p 新模型探索计划（尚未实现，不是收益）

当前控制必须是407源的prefill INT8推荐版，正式5586.28、Math102/105；不要沿用359源 sampler_exp_service老控制。新profile应继承 tianshu_prefill_int8_service_20261021/candidate_profile.prepare(root,'candidate')，两臂都保持NN decode/prefill INT8/K8V16/全部默认与原13个CLI参数，62项显式环境只改PYTHONPATH到新hooks_control/candidate。parent.audit_runtime必须用dict(state,mode='candidate')，保留407父审计全部条件，新增sampler调用/源码/预热/配置/真实路由记录。新源数量由CPU prepare实际计算、两臂闭包完全相同，不硬套旧359/407/445计数。

前置：补真实FP32 contiguous stride130560组件；tianshu_topp_launch是stride130567压力布局，不能当真实layout。旧topP warp16/8 bitwise native gate false必须保留，不能改成通过。p≈.95数学诊断无rank inversion，原生本身也有FP64 nucleus差异，warp16对原生截断分布最大TV .00186977/平均.0000140734（1208有效合成行），不证明任意输入。用户允许严格>95%质量判定，不应把自己设置的逐位门槛当官方限制：后续是明确允许FP32归约差异的新模型候选，须新实际logits/小A/B/新Math105及正式结果；不复用旧失败来宣称等价。

候选只改变原JIT launch num_warps=16。完整原始算法、BLOCK8192/TRUNC4096、温度/top_p/rejection/RNG一律不改。进程作用域包装native module._topk_topp_kernel，委派原__getitem__；__getattr__保留原fn等inspect接口。仅CUDA FP32词表130560、B8..64、TopK disabled、TopP enabled、stride1=1且stride0=130560时选16，其他情况原配置。不要在Python读取p GPU值或增加同步；算法原有p处理保留。C8覆盖官方Math batch8，C64覆盖吞吐。实际logits的layout、dtype、p/k metadata/route及compiled binary需记录验证，组件不是真实模型证据。

预热必须在真正原生memory_profiling_fl内部、原context yield之前执行，两臂预热同样W4/W16与B8..64。实际WorkerFL在with内部先读取profile_torch_peak（line570）再profile_cudagraph_memory，退出context后用该局部peak覆盖profile_result.torch_peak_increase（line586）；因此包装context在yield之后预热会漏计新临时峰值，不能采用。正确顺序是with original_memory_profiling_fl(...) as result: warm_both_variants(); yield result。此时warm缓存/临时峰值先进入native stats，后续原profile_run与profile_torch_peak读取自然包含它们；不增加profile_run调用栈、不reset peak、不改KV预算。需要按原native context before/after快照、实际availableKV和buffer容量复核，所有新增持久buffer也在同一个原生计量范围。

warm用私有CPU generator或确定性数据，不消耗默认CPU/GPU RNG；输入/暂存Torch tensor及时释放，同样代码两臂执行。原生既有buffer容量至少min(SM,64)；若需新增缓存，只能在真正原生profile内。记warm前后默认RNG、retained memory、peak与native availableKV，临时warm tensor不在KV规划之后新增。W16实际profile内预热，真实request配置仅在完整父capture_model链返回之后激活；这样控制保持4，候选16，目标模型Graph/prefill激活顺序均保留。sampler在Graph外执行，不能伪称目标Graph覆盖W16。

真实服务需记录MODE、originalJIT SHA及源绑定、default warp4/component同binary证据、actual request calls的B/stride/warp、kernel metadata/binary、profile/capture顺序、默认RNG不变、原capture返回值不变。暖身调用与实际request命中明确区分，计数不是Graph replay计数。若源复合hook失败，真实代码修复必须新实验目录，旧失败保留。

复用父run_service_trial模式：原始命令、4K/1K C64 N64三轮排首轮，输入seed每轮变且跨臂配对，保留所有response/usage、canary复现、warm/autotune/FutureKV/抢占/native memory/完整父审计。只要有明确整模型收益再跑新Math105严格>95，失败不抽样重跑；再正式N256 4K。不要为本组件常规跑16K，也不拿28-35%过滤耗时下降换算整模型提升。当前没有任何新TopP模型score。

约到2026-10-22 00:00，北京23:58硬停。仅容器/GitHub，不碰用户dirty2文件/沐曦仓库，不启动子agent；每轮报成绩。所有GPU启动过的目录源冻结，恢复metadata独立CPU目录。
