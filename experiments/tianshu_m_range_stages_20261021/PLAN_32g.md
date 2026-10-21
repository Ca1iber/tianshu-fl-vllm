# 按环节、M范围探索重排与融合

用户最新指示：先完成当前450源融合候选的C64补充Math与正式4K，再做本轮。现在仅CPU准备；没有任何新的GPU任务。当前watcher249716及其子进程未全部终态前不能启动本轮。

参考沐曦flagos-contest ed3bcfe中vllm_fl/ops/packed_bundle.py：_BUNDLE_ROWS=(576,1152,1664,1920,2048)。这五点控制MLP和QKV/O projection，但LM head另有(1,8,32,64)条件。不能声称所有环节恰好都用五点。源位于天数容器bare只读参考，不修改沐曦仓库。FlagGems packed来源8eb57f17，两个源仅适配vendor guard，所有设备函数AST必须相同。

天数当前基准必须采用其实际路径：GateUp decode M8/64为NN INT8，候选M64实际融合；prefill已预热24个M128..4096形状走NT INT8。Down/QKV/O仍BF16，LM M64 INT8、M8 BF16。其它M先按实际policy和图记录确定，不用BF16慢对照冒充现有INT8基准。

第一阶段逐角色矩阵，探索采样M=[1,8,16,32,48,64,96,128,192,256,384,512,576,768,1024,1152,1536,1664,1920,2048,2560,3072,3584,4096]。QKV/O/Down/GateUp覆盖实际prefill和decode形状；LM只测实际计算logits的1..64及诊断128，不在整段4096上无意义生成全词表。另从当前正式4K原始route统计补实际命中点。

分离并记录：静态W重排的时间与新增bytes；逐步A重排或A量化时间；相同tile下普通W与packed-W GEMM；A+W重排GEMM；GateUp GEMM+原激活与融合；Down独立GEMM；完整GateUp/SiLU/Down链（含activation layout转换）。QKV和O的形状/配置独立，LM单列。INT8 GateUp额外比较NT/NN/NN+融合，使用当前真实同一packed/scales，不额外改变精度。不能把重排、tile、融合联合收益都归因于重排。

通用INT8融合仅放宽host M/BM/warps约束，原_fused设备AST和native SiLU标量不变；禁止已知错误single_dot。BF16 MetaX移植设备AST不变，GPU实测后再决定天数配置，不假定旧五点或任意相邻范围都有收益。

每一候选先完整输出对照、两次改变A与W/scale、独立CPU选行/列/INT64或BF16-P语义核查、真实Graph交替计时。实际权重和真实shape优先；组件首筛后用42层权重轮换核验缓存偏差，再选少量候选进4K/1K C64 N64小测。静态权重额外显存必须计入原生profile并报告KV容量变化。

第二阶段只在有收益段加密：从相邻采样点内部、边界与padding前后补测；实际M可达点全部核验后才生成按role分段的白名单，收益不稳的区间保留现有实现。查表开销、输入重排和端到端MLP耗时都计入。不能仅凭两个端点就宣称整段获益。

第三阶段4K真实服务A/B（原命令、相同输入/输出/请求数与随机seed），完成一轮就报数；稳定有收益才跑新鲜Math>95%并正式N2564K，不例行16K。代码、报告与关键结果GitHub，原始张量和完整日志留容器。最晚2026-10-21 23:58北京时间退出GPU。

本目录目前是源和计划准备，不是已经完成GPU验证的实验，也没有完成GPU driver/最终CPU冻结清单。后续准备完成后才能冻结启动。
