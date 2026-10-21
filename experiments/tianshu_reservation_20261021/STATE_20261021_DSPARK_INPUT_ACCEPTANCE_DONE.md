# 2026-10-21 10:05北京：投机输入契约+接受率诊断完成

预约2026-10-22 00:00，硬GPU停23:58/15:58Z。SSH/tmp/tianshu-20261021-control 04:27建立，16:27前刷新。当前GPU空闲68MiB，无任务；输入probe203109和接受率bash203675均terminalZ，服务203789/Engine203838已清理。两项GPU+CPU结果完成。目标仍active，不标完成。

推荐保持tag tianshu-original-command-4k-prefill-int8-20261021：正式4K5586.28/1117.25output、TTFT3780.63、Math102/105。8192formal5608.40(+.396%但TTFT4406.24)不采；static+.254%/KV边界+.0543%不采。新组件不当模型成绩；旧16K3885.91不是本新版本。此前多查询26场景报告已c02bc5d，语义复核4cb4b67推送。

新inputcontract experiments/tianshu_dspark_input_contract_20261021：16+gate17源，312 CPU实际安装版fit边界/aux方法，GPU10fixture×2phase×4提案数1/2/3/7×2sampleview=160调用全过。原DSparkhost和安装DFlash/原始prepare kernel，无模型/KV写入。20fixture覆盖ctx1/3/17/256/257/513，B8混合/B64，拒绝、跨页row stride、合法最大pos131071。全部原输入/输出guard/RNG保持。CPU结果dirtianshu_dspark_input_contract_results_20261021逐值复核111424metadata，rawSHA4a10cdeb...。newpolicy.py仅DSpark架构按实际七槽限长，普通DFlash不动；旧nSpec+1对1/2/3少算，对7多算1。此缺口不解释4K低接受率。2d98839/2c4f541已push。

新acceptanceprobe experiments/tianshu_dspark_acceptance_probe_20261021：GPU33primary sources(不是full transitive closure)、CPU gate锁定sha c2327b...。CPU tokenizer显式return_dict=False，实际dataset.level数字3，prompt长度[55,20,28,42,33,56,21,36]，random逐条长度匹配，另4Krandom8。C8输出128 T1/p.95/ignoreEOS固定，3组各16暖身后ABC/CBA2轮，72响应全部usage正确。actual request hook证明42+5、BF16共享LM、aux[2,11,21,31,40]、七query两proposal、standard probabilistic/denseq、正FP32阈值、rejection ratio/kernel未改。新七槽policy服务请求命中已证，maxlen边界服务未实测(只有CPU和prepareGPU合法边界)。

结果natural每verify接收1.711640/1.785326，weighted1.747989(746verify/1304accept)；equal-lengthrandom .201176/.273183，weighted.236044(1648/389)；4Krandom .162286/.213604，weighted.187391(1713/321)。相同内容方向2轮支持workload effect，小8prompt+128输出不能泛化/等同oldC64/output1024接受.5944。无新吞吐/Mathscore。全部CPU独立parse raw counters/72response/promptsha/source snapshot proof tianshu_dspark_acceptance_probe_results_20261021/verified。原GPU/py/sh/gate不可改。Client_result rawSHA7b83d992...。40b9b85准备点已push，结果本次待提交/推送。

下一方向实际sampler top-p：native C>=8使用topk_topp_triton.Qrita，不是PyTorchfullsort。实际top-kNone,p.95，单CTA per row，多遍pivot/outlier scan，baselineBLOCK8192/TRUNC4096/defaultwarp4。可测试相同JIT AST/仅num_warps2/8/16、或blocksize探索(后者初始统计和rounding可变，需单独数值证明)，先fullmask/native/CPU双oracle+真实input+RNG不变/全guard、再hotGraph过滤组件/含noise完整服务小对照。有>2%稳定才Math/正式。旧sampler融合/Philox服务仅+.33/+.368%，GraphV1RNG失败、V2完整staged3.491→3.645ms负4.39%，不可重复误当新收益。当前完整sampler约3.49ms(旧profile测量)而非猜测10ms，不先声称主瓶颈。

另一保留可行方向Spec target M192 GateUp、LM128/192 INT8 component，现prefillgroupedkernel支持任意M但profile仅aligned128 M(192fallback)，需要新真实M数值+实际BF16基线/寄存器/时延，47KV、scratch及43bank均nativeprofile计费。当前407NoSpec不能直接SpecNone关掉；所有42因果假设需显式42+5noncausal验证。随机接受低限制收益但不等于框架无空间。

禁止本地副本/子agent；仅容器/GitHub。两处用户gateup_silu dirty不动。Ops4cdc3b7不改。每轮报告、60秒更新。直接shell/git，Python用于数值/分析不包装文件操作。新源GPU启动后immutable，真正代码变更新dir，元数据恢复独立CPUdir。观察超时不重启，先实际PID/state。
