# NN GateUp+SiLU融合：新权重布局组件，模型接入另审

正式407源/62env版本5586.28、Math102/105不变。旧NT布局融合组件约13.67%收益，但MLP opaque V2/V3实际服务慢于原参考；不能重用旧NT时序为新NN收益，也不重复原opaque fallback方案。

组件仅actualM8/64，连续与行stride2055四案例，均不补到32/64。原生对照为当前真实NN kernel.quantize和NN GEMM选优配置（M8 bm32/M64 bm64、bn256/bk128/w4/s1）加实际FlagGems SiLU标量；排除静态权重转置，双方包含真实A量化。融合two_dot / single_dot各bn64/128×warps2/4、bk128/stages1，32组形状配置。single_dot加载连续gate/up两半、INT32 dot后reshape/trans/split；two_dot共用A分别dot，均恢复BF16投影舍入再调用同一FlagGems _scalar_fn。不是改sigmoid近似。

CPU先AST语法/原407源hash/current NN真实选优配置/成对列覆盖与split顺序/INT32无溢出界限；不初始化CUDA。真实layer0权重、packed/scales完整签名GPU进程内CPU准备时与合格版比对，W以NN物理布局复制，逻辑值不变。

各组3次原地改变A/W/scales并重放同Graph，实际计时版本全BF16输出必须逐位等于原生对照，完整输出保存供离线重读。诊断版本另外核全INT32和BF16投影，选行/列独立CPU INT64与BF16/SiLU参考，离线再从checkpoint取原行。边界guard、完整输入/权重及RNG不变都检查；7次轮转顺序、每组20replays。编译资源/编译错误单独记录，不作为数值通过；数值失败停止并保留。

这是声明的FlagGems激活组件，不宣称它与原模型Torch编译的activation逐位相同。GPU组件若有收益，再处理真实FX局部融合和新Math；compile_sizes8/64支持单点静态范围，动态1..4096应保留。原后端占用pre_grad_custom_pass做IR功能化，不能直接覆盖；post_grad支持自定义InductorPass和pass_context.compile_range，需核真实decomposed graph再精确匹配。历史dc30c90ffa计算图含QKV测试包装，不是原407源证明，仅作结构参考。尚无新模型hook/FX改动，不启动服务或Math。

GPU前冻结源码/CPU_GATE/PLAN；单进程，1800秒与23:58硬停，GPU后独立CPU核验。后处理若只有元数据问题另CPU恢复，不重跑GPU。代码报告关键gate进GitHub，完整tensor/log/snapshot只留容器，本地不留副本，原用户dirty文件与沐曦不改。
