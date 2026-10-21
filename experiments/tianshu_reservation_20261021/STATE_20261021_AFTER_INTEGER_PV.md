# 13点后：整数PV完整Attention不采用，下一项QKV prefill INT8

预约10-22 00:00止，GPU23:58硬停。推荐仍tianshu-original-command-4k-prefill-int8-20261021 / 521c154，407源，正式4K5586.28total、1117.25out、TTFT3780.63ms、Math102/105。TopP失败99/105不重测；Down SplitK与grid-only无收益已归档。

新整数PV full GPU十场景210路径全部完成，独立CPU恢复后2184向量/1104整数向量/302592writer V与K/42bitwise闭环；两项理想attention原容差失败保留。4K/C64当前.539900ms，最快整数PV1.450207ms，明显退化，不接模型。原verify只在2D segments元数据谓词失败，原件不改，新CPU recovery不重跑GPU。报告tianshu_k8vi8_pv_full_20261021/REPORT_32g.md，最终gate在tianshu_k8vi8_pv_full_cpu_recovery_20261021。原GPU230469及原CPU231242终态Z，恢复CPU231327已写complete gate，GPU空闲。

下一项只读审查已测真实QKV大M INT8组件。partial QKV gate完成9M×20配置×3phase，540完整GPU/CPU整数检查；父双投影整体O对照失败继续保留。QKV M4096实际BF16 .492038ms→A8W8 .428116ms，需判断42层220631040B新增payload是否值得接现行407源。只考虑已测prefill M，decode不改；需原生DeviceMemoryProfiler计量、真实路由/capture/权重审计、新Math>95，不能把best-of-20局部14.93%当模型收益。O的三项native参考失败未解决，不在本候选叠加。

所有代码/报告/关键结果容器+GitHub，无本地副本。两份用户dirty gateup_silu不碰，FlagGems clean4cdc3b7，沐曦只读；无子agent。SSH master /tmp/tianshu-20261021-control，16:27前刷新。
