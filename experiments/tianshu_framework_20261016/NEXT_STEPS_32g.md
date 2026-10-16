# 天数框架探索续跑清单（2026-10-16夜间检查点）

正式最佳4093.81 total tok/s、818.76 output tok/s、TTFT4309.35ms，独立Math102/105。当天框架候选均未新增采用。保持最佳tag tianshu-original-command-4k-geo-20261016→ca0e22d；不修改沐曦仓库。只做4K筛选，明确收益后重跑fresh105及正式N256。

第一优先：小批次采样集成。B1..7 native sort→既有GEO主体组件耗时减少29%至79%（普通分布），16阶段/160事件/62源已完整备份核验。还没有真实模型、采样RNG或服务证据。先做有界路由，unsupported全部回退，真实采样API及每请求RNG canary，再N64不同输入对照。适用范围是尾部小批次，不能把组件比例搬成C64收益。原件和报告见tianshu_tiny_topp_20261016/results/20261016/micro_32g。

第二优先：投机prefill动态M。固定index后的ngram试验R1/R2零调优，R3仍出现M2039四种GEMM，合计2.31秒。实际75/hybrid guard关闭。可以研究仅maxQ>2、eager prefill的padding，使2039→2048，但不能全局解除spec guard、修改真实调度token或把混合验证伪装为uniform decode。暖点是否真实加载、实际padding日志、GEMM cache miss都需要逐项核验。

第三优先：实际投机图覆盖率。已冻结model_runner.py（service_pilot source_snapshot中的安装源）第3871至3889行_is_uniform_decode，条件为maxQ==uniform_decode_query_len且num_tokens==maxQ*num_reqs；第875行uniform_decode_query_len=1+num_spec_tokens。因此1-token投机全Q2满足、混合Q1/Q2以及全Q1不满足。第3936至3948行把该bool交给dispatcher。先做低开销路由计数并区分profile/capture/请求阶段、纯decode/验证/prefill，测量真正丢图比例。当前只有组件/unique route及Python调用计数，不代表GPU每步图重放计数。

若确认值得做，再设计可变cu_seqlens固定容量Q≤128/Req≤64图；attention正确不代表模型图正确，必须逐层验证input_ids/positions、query_start_loc、真实num_actual_tokens、logits_indices、KV slot -1补位和下一步接受数修正。现有model_runner第4094至4104行生成padding slot，4341至4354行控制pad_attn；不能只改uniform布尔或force_uniform_decode。保留L0数值兼容及正FP32标准拒绝，禁止合成接受率和退化FP64阈值。

第四优先：投机与异步调度。当前真实model记录async=False、async_spec=False，非投机控制async=True；本轮2670.589比近期4037.683慢33.858%包含框架策略变化，不能全归因kernel。原生async/spec兼容性、接受token与host缓存更新的依赖要先核验，不能直接强开。已有非投机async-off对照约-6.28%，仅能作为线索，不可外推出投机收益。

低优先：KV软件8位浮点u32在attention组件上仅省约3%，writer、cache生命周期、prefill和新质量均未完成；若继续必须完整算上这些成本。NHD/HND、页表同步、静态INT8、请求位置准备融合和prefill cadence都已有完整证据，避免重复刷微小组件收益。

记录原则：原件、输入种子、实际env/flags、源码SHA与CPU/GPU对照分别保留；逐轮向用户报成绩。每个新版本先小规模4K，保持原参数；只在明确端到端提升时投入新Math/正式。新时段必须获用户确认，本次GPU截止2026-10-16 23:58+08，预约到2026-10-17零点。
