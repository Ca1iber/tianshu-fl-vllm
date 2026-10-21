# 2026-10-21 10:22北京：TopP两布局组件完成，下一步真实407源服务A/B

Active goal继续优化到2026-10-22 00:00，23:58/15:58Z硬停GPU。当前GPU空闲68MiB，TopPstrided208397/contiguous208814 terminalZ；全部CPU验证会话62293/59408已结束，不重启。SSH socket /tmp/tianshu-20261021-control04:27建立，16:27前刷新。无子agent。容器/GitHub唯一存放，用户2个gateup_silu dirty不动，Ops4cdc3b7不变。

当前推荐tag tianshu-original-command-4k-prefill-int8-20261021：正式4K5586.28 total/1117.25 output、TTFT3780.63、Math102/105。8192formal5608.40(+.396%但TTFT4406.24)不替换；static+.254%/KVbound+.0543%不替换。当前新16K没有，旧3885.91旧profile。前次状态STATE_20261021_DSPARK_INPUT_ACCEPTANCE_DONE保留。

本轮完成：DSpark实际输入准备160调用/111424metadata CPU值全过，312原生fit/aux CPU边界。newpolicy七slot修正已在后续服务实际hit，普通DFlash不变。接受率probe33primary(非传递闭包)、C8输出128、3组ABC/CBA各2轮+warm，72响应全CPUverified。自然Math weighted1.747989(746verify/1304accept)；等长random .236044(1648/389)；4Krandom .187391(1713/321)。保持standard真实denseq概率/正FP32阈值、42+5 BF16共享LM/aux[2,11,21,31,40]，不作新Math/吞吐。结果51499f9已push。随机workload低acceptance有证据，不说所有移植数值已排除或投机绝对无空间。

新TopP方向：nativeC>=8本来已Qrita，不是full PyTorchsort。旧采样融合/Philox只有+.33/.368%模型、GraphV2整体−4.39%，不要重复。新只原数值JIT不变BLOCK8192/TRUNC4096/TopKFalse，warp2/4/8/16。strided目录tianshu_topp_launch_20261021 rowstride130567，12cases×3phase×5modes180全输出，10frozen源，rawSHAda35fa29...，CPU全部846028800float重建SHA/1272FP64nucleus行/420timings。W16有28-35%filter耗时减，非模型，bitwiseNative不全等。完整report/GATE/keytimes/cutoffsummary/TVsummary已6ffa2a0push。p95有效1208行W16diff32/592elements无rankinv，maxTV.00186977 avg.0000140734。原生同样对FP64有330行maskdiff/5行massbelowp/maxshortfall.000760746；pother异常另保留，绝不改判旧bitwise gatefalse。

因strided非production又建真实contiguous目录tianshu_topp_contiguous_20261021，所有workrowstride130560且首尾完整guard行，独立verify CPU额外断言layout；不改oldsource。新12/180/846028800/1272/420/10frozen全过，rawSHAb4f8818c...。C64 native→W16：normal1 6.588740→4.276324ms(-35.10%time),normal4 4.322644→3.108076(-28.10%),peaked1.696111→1.131965(-33.26%)。W4显式与native同binary/全输出相等，W2/8/16仍部分maskdiff。单线程regs193→70，spill都0，不能claim总CTAregs/occupancy。是过滤+restore hotGraph组件，不含RNG/完整Sampler/模型，当前未采用。

CPU布局对比读取36对BF16输入/p完全一致、180对完整floatSHA+mask全相同，因此旧cutoff/TV数值精确适用这组输出(有LAYOUT_NUMERICAL_COMPARISON.json)，不用重复GPU/数学重算。不泛化任意输入。新results目录tianshu_topp_contiguous_results_20261021保留独立verify/compare代码和keyJSON/CSV；逐行CPUideal大文件移容器benchmark-results/20261021_topp_contiguous_CPU_tianshu，不重复上传。旧report和此report/下一方案本次提交待push。

下一步请直接实现新的服务A/B，不再停留重复同组件。详细NEXT_TOPP_SERVICE_PLAN_20261021.md(已按实际WorkerFL源码更正预热顺序)。继承tianshu_prefill_int8_service_20261021/candidate_profile.prepare(root,'candidate')两臂均当前407原13CLI/62env/全部K8V16+NN+prefillINT8；新hook仅samplewarp，parent.audit_runtime用mode='candidate'，保留全部父audit，不沿旧sampler359源父版本。新源计数CPU实际算，不硬套。候选明确允许归约数值变化，用户允许准确率>95，不能把自己逐位门槛当官方禁令；但旧失败仍false，新profile小A/B+新Math105>95+formal独立取得资格。

预热计费关键：actual installed /usr/local/lib/python3.12/site-packages/vllm_fl/worker/worker.py 6cf62f60ca8ea12b7c494f0009fa733f33270a73e014cdbd4a45cada787c032a的determine_available_memory native resetpeak558，with memory_profiling_fl562；profile_run566；with内读取profile_torch_peak570；graph估算584；with外586用这个旧局部peak覆盖result。因此memory context yield后再warm会漏计临时peak，不能做！建议wrap原memory_profiling_fl：with original(...) as result: warm_both_W4_W16_B8to64(); yield result，warm在yield前且原context内，之后原profile_run和nativepeak自然包含；不加profile_run包装，避免已有engine_hooks inspect.stack()[:8]漏determine_available_memory。全部新增persistent sampler cache/临时/模块纳入原nativeprofile，两臂warm相同/defaultRNG不变，禁止reset峰值/改KV预算；CaptureModel完整父链返回后才激活候选W16，sampler仍Graph外，不能说targetGraph覆盖W16。

新Router委派originalJIT __getitem__/__getattr__fn，eligibility CUDAFP32/130560/B8..64/TopKFalse/TopPTrue/contiguous，其他原fallback；不Python读取p GPU值或同步。保留原p/duplicate/error处理，改numWarps16。实际服务记录layout/真实requestB/warp/compiledbin/warm vsactive、RNG/原capture_return/nativeKVmemory。C8覆盖Math8、C64性能。先4K/1K C64N64三轮排首轮，稳定明显才新105 Math严格>95/正式N256，不常规16K，失败Math不抽样重跑。

所有GPU启动过的目录source/CPU_GATE frozen，真正代码fix新目录；CPU元数据恢复另目录。观察超时不重启，查实际PID；每完成轮报成绩，60秒更新。目标仍active，不因阶段完而complete。
