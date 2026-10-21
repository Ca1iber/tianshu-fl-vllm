# 13:32 QKV V2候选启动中，仍无新最佳成绩

预约10-22 00:00止，GPU23:58硬停。正式推荐不变：tag tianshu-original-command-4k-prefill-int8-20261021 / 521c154，407源/62env/原13argv，4K5586.28total/1117.25out/TTFT3780.63ms，Math102/105。TopP fresh99/105失败不重跑；Down SplitK、grid-only及完整整数PV皆负收益，报告已GitHub。

整数PV原GPU十场景210路径完成；原CPU verifier在末尾2D segments检查失败，独立CPU recovery只修正原生2D使用1而不是未使用入参16，不重跑GPU。2184CPU向量/1104整数向量/302592writer V与K/42bitwise闭环；两项理想attention原容差失败保留。4K当前.539900ms vs最快PV1.450207ms，拒绝接模型。完整报告与恢复gate均已44298f3推送。

QKV prefill INT8 V1在加载模型前因新routing.py遮蔽原pipeline routing.install失败，零轮/候选未启动。原失败和源保留。独立V2专属qkv_prefill_routing与weight_pack，CPU检查原routing解析正确，全部42层220200960权重独立NumPy量化、4700tile、139296路由检查通过，V1/V2完整bank签名逐字节相同；432源，已96e78f0推送。

V2冻结目录tianshu_qkv_prefill_int8_service_v2_20261021。runner232213；runner根/workspace/benchmark-results/20261021_qkv_prefill_int8_service_v2_pair_runner_tianshu。control已complete，三轮5270.25444（排除）、5480.51822、5482.10410，TTFT12.54927/10.37352/10.41515s，全部tune0；后两轮均值5481.31116，spread约.02893%。已逐轮向用户报告。candidate正在starting_server，server233355、EngineCore233513，42banks已处理，尚无candidate成绩/新Math。原control server232223/core232364已退出；不要重启232213任务。

QKV只有已预热prefill进入新INT8，decode调用原apply。该方法委托记录不证明原编译上下文/实际BF16kernel逐位相同；新opaque wrapper可能有共同成本，故在候选出分前增加独立exact-reference计划。CPU gate已证明原407全部source是候选432的完全相同子集，真正未包装参考使用原frozen prefill_int8_service/run_service_trial.py mode=candidate。目录tianshu_qkv_prefill_exact_reference_20261021。仅V2 SCREEN预声明门槛通过，才启动run_reference.sh，然后候选还须相对真407 >.5%且超两臂极差/spread<.5/TTFT<=1%/tune0，才准备新Math；若V2失败或无收益，跳过。不要拿5481这个包装后control单独宣称比当前推荐快。

质量尚未准备新controller；只有值得时以现行prefill_int8_validation（C8 Math/C64 formal）为基底接新QKV实际路由/权重审计。不能照搬TopP validation的采样candidate条件或438source要求。最终432runtime与fresh105>95%绑定，再N256四轮排首轮且绝对超过5586.28才可采用。无新完整16K。

代码/报告/关键JSONCSV仅容器+GitHub，无本地副本，不动用户dirty gateup_silu两文件，FlagGems clean4cdc3b7，沐曦只读，无子agent。参考裸仓在/workspace/reference-repos/metax-latest/flagos-{contest,ops-lib}.git，HEAD master为空，查看refs/remotes/origin/exp/packed-bundle-service-20261007；latest refs ed3bcfe / 8eb57f17。SQL reflect_one已在当前版。LM NN布局也早已测负收益，别重复当新方法。SSH master/tmp/tianshu-20261021-control，16:27前刷新。
