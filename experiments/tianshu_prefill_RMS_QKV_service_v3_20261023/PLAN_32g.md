# r3基础上的prefill M4096 QKV norm+A8融合：接入开发阶段

当前合格基准仍r3的903项模型来源与正式5950.75 total tok/s。刚结束的RMS64+LM组合仅短测+0.2535%，不采用；本版本不合入该组合，单独接入新prefill完整QKV链组件（b7a4990）：M4096图延迟减少4.81%—6.30%，eager减少4.95%—6.50%，18数值/180计时及完整独立CPU通过。M512带残差graph16仅2.09%—2.16%，且eager包含更多提交开销，本次先只选证据更强的M4096，不扩展既有量化精度范围。

计划在原compile_sizes=[64]之后赋予新的[64,4096]列表，保留旧组件安装时的原列表快照并明确其unchanged声明仅限原钩子作用域；实际新增范围由新记录和43个静态4096图直接审计。仅[4096,4096]内42个QKV前IR norm替换；首层无残差返回QKV，后41层返回QKV及新BF16 residual，其余通用图和M64图保持前后字节不变。新FX节点只写val元信息，避免旧eager_input_vals参数数目不匹配。

复用原84组NN weight pack、原BF16参数、native _int_mm与原dequant CUBIN，不新增pack/global activation workspace。通过固定Torch2.10 OPDEFS的原_init_fn闭包读取真正原banks/record/save，原文件字节不改；新路由若进入原兼容索引，只统计真实相同native MM+epilogue，并明确standalone_A8_quantizer_executed=false、norm+A8融合和新来源。不能冒称原QKV custom_op或旧独立量化器被执行。

新的scratch必须在真实determine_available_memory/Profile窗口内用实际原bank0无残差和bank2有残差的显式zero probe计入峰值；初次general profile可能尚未执行静态4096路径，不能据此宣称新路径已被profile。probe独立标记、保持RNG/原权重，且manual KV bytes仍None，全部旧memory/profile审计保留。真实静态4096暖图/运行42层和新norm/dequant CUBIN、全部旧KV/scheduler/sampling/Down/O/GU/QKV其它M/LM守卫必须独立核验。

本目录已实现parent绑定、policy、静态4096 FX pass、真实内存profile探测、服务来源冻结与新路径审计；初次CPU准备已保存在CPU_INITIAL_PREP_GATE及原准备目录，尚未启动模型。待GPU前完整冻结后，新的C8/C64各105题严格>95%、全部独立运行/来源审计通过，才与r3同期N64×3排首轮短4K；收益>0.5%/波动<1%/TTFT<=1.01后才原N256×4完整4K。不运行16K，不自动采用或发布。15:58Z硬停，原始数组/日志仅容器，代码报告关键结果GitHub。

v2接入修正：v1首次Worker初始化在旧NN融合policy的绝对`from fx_pass import FusionPass`处误命中新目录同名模块，尚未载入权重或发出Math请求。原v1代码/失败日志保留；本版新pass改名prefill_qkv_fx.py，且不将新服务目录加入sys.path，避免影响旧绝对导入。数值kernel及原r3来源均不改。v2重新做CPU冻结与新鲜质量验证。

v3修正：旧RangePass/NativePrefillPass仅支持[1,4096]，新增[4096,4096]若直接调用旧链，会留下GateUp linear+SiLU而跳过正式native GateUp。v2在旧GU真实profile守卫处停止，未答题。本版在新静态范围独立复用原range_match，直接形成相同tianshu_native_prefill.fused_or_original，保留其真实原pack/MM/epilogue以及所有profile/CUBIN守卫；其余两个原范围仍逐一调用原pass，原86份pass记录真实保留，新范围另外43份独立审计。新pass须证明42个GU恢复、42个QKV融合。未伪造旧pass执行记录，未改原903来源或数值内核。
