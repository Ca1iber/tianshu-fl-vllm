# 天数：原始命令CV下的M2048 GEMM与strictTopP16K组合

本候选只组合已有数值证据的两项算子配置，在当前原始命令控制下重新测量，不相加不同历史配置的小收益。QKV与其他M保持原GEMM，M2048 O/down采用256/256/32/w16/s2/group1，gate_up为group8，无输出拷贝/重排/量化；TopP纯P的B8..64采用严格修复源029d、tile16384/8192/w16，其余回退原生TopK/小batch。原采样器/RNG、CV/KV16-w1/S8/nativePrefill保留。

两组serve argv逐项等原始官方命令，default budget2048/maxseq256/asyncTrue/stream1，实际75预热点。只增加3个实验环境项（总38），差别仅TopP服务模式与M2048开关；目录按组映射。不设置或隐藏调度参数。

真实CPU启动链控制/候选均检查唯一CV、Worker、TopP、SQL finder；候选只额外注册既有GEMM finder，不重复原基线。未导入torch。TopP完整旧数据重算4224语义行/420events，fixed16k严格1e-6通过；GEMM48case中40通过8失败原样保留，只选择三个明确通过配置，重算560events。新组合尚无服务/Math/正式结果。

3+3短筛：每轮4K/1K/C64/N64、temperature1/top_p.95，逐轮使用不同输入seed、跨组配对；预排首轮，保留完整384响应。门槛>=2%且两计分轮不重叠才考虑独立确认/新Math/必要正式4K。C1 seed777 canary一致只验证回退，不代替B64质量。M2048与TopP路由记录每shape首次观测，可能含启动warmup，不冒充逐请求调用计数；端到端收益依据真实响应总时长。

运行：`python3 experiments/tianshu_gemm_topp_20261016/run_service_pair.py --output NEW_ROOT --stop-at-utc ISO`。两组串行，原始源码冻结、原始命令和默认调度/actual38env/CV/75warm/三GEMM路由/严格TopP源/零preemption均作实际审计；截止17:58local。当前正式3992.97/Math103保持，未启新Math或正式。

条件后续脚本已准备，尚未启动：run_confirm_pair.py --screen-root SCREEN --output NEWROOT --stop-at-utc ISO，实际screen>=2%且计分轮不交叠才允许独立5+5（预排前2轮）。run_validation.py --confirm-root CONFIRM --output NEWROOT --stage math|official --stop-at-utc ISO，confirm同样>=2%且3计分轮不交叠后重放实际候选38环境/54源，再冻结2个质量脚本，共56源；official另需--math-root NEWMATH且105完整新Math>95%。两项都由用户时段显式截止，不读取旧全局deadline。离线verifier分别支持3+3、5+5、Math和正式；历史105解析103仅CPU解析兼容性，不借用旧质量。
