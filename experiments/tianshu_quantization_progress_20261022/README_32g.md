# 天数量化探索进度，2026-10-22

截至北京时间13:29，新原生INT8预填充的全新C8与真实C64均101/105；C64实际M4096全42层覆盖和完整审核通过，独立CPU复核完成。正式4K已进入运行流程。**尚未产生新的正式N256成绩，正式最好仍是5695.95 total tok/s。** 所有代码、报告与关键结果均在天数仓库；原始数组、日志和完整响应仅留容器。

## 已确认的结果

|范围|事实|状态|
|---|---|---|
|第一阶段W4/W8新权重组件|9版、300配置、900输入相位均经独立数值审核；本轮最快W4A8 Down仍比同进程BF16多28.1%耗时|保留负结果，不接入|
|A8倒数边界修正|有限BF16 GPU域q/scale审核通过，测试6种shape都比原量化器慢|不接入|
|原生INT8 M64 GateUp|完整量化+整数GEMM+BF16舍入+SiLU，最快0.066165ms，对照0.063122ms|不改变原M64路径|
|原生INT8 M512/M4096，全部42实际MLP bank含Down|每层完整MLP逆耗时改善6.16%/18.07%，252次变化输入/bank检查与独立CPU审核通过|仅组件证据，不冒充整模型|
|真实模型4K/1K，C64/N64，两臂各三轮排首轮|控制5600.66、候选5721.43 total tok/s，+2.1563%；TTFT-5.6603%；候选后两轮差幅0.09224%，全部384个计时响应成功，无就绪后autotune|完整运行时审计和CPU重算通过|
|全新官方C8 Math105|101/105=96.1905%，105请求成功、0接口错误；预填充实际全部BF16，INT8命中0|原GPU audit_failed保留，独立CPU范围复核完成|
|全新真实C64 Math105|101/105=96.1905%，新M4096原生INT8真实全42层、M64融合Graph与LM64 INT8通过；105题自然stop无截断|原GPU完整审计complete，独立CPU复核完成|
|全新官方4K N256四轮排首轮|两份新质量全部通过后才运行|正在运行流程，尚无新正式成绩|

## 证据入口

- [第一阶段完整负结果](../tianshu_quantization_survey_20261022/REPORT_FINAL_32g.md)：该报告的“收尾”只属于当时的第一阶段，后续原生INT8阳性实验见下方，不用其历史结论覆盖后续发现。
- [原生INT8预填充单bank组件](../tianshu_native_int8_prefill_v2_20261022/REPORT_32g.md)。
- [全部42bank含Down的独立holdout](../tianshu_native_int8_mlp42_20261022/REPORT_32g.md)。
- [真实服务短测报告](../tianshu_native_int8_prefill_results_v2_20261022/REPORT_32g.md)与[独立CPU筛选门槛](../tianshu_native_int8_prefill_results_v2_20261022/SCREEN_GATE_32g.json)。
- [完整独立质量与正式评测计划](../tianshu_native_int8_prefill_validation_20261022/PLAN_32g.md)。
- [新C8按实际范围复核报告](../tianshu_native_int8_prefill_math8_recovery_20261022/REPORT_32g.md)：原失败保留，不把C8未命中路径冒充INT8质量。
- [新C8与真实C64独立质量汇总](../tianshu_native_int8_prefill_quality_results_20261022/REPORT_32g.md)与[QUALITY_GATE](../tianshu_native_int8_prefill_quality_results_20261022/QUALITY_GATE_32g.json)。

新方案复用已有W8 NN布局和A8量化，只对M512/4096替换计算实现，原模型checkpoint与架构不变，不训练或微调。两臂877份冻结源相同、原正式471份精确子集；原13项serve argv保留，62项继承环境加明示TRITON_CACHE_DIR共63项实际观察。新192MiB临时INT32工作区在原生profile内实际覆盖42层并计入显存预算；M8/M64、LM、其他M与KV逻辑保持原路径。

首次启动目录冲突、单轮smoke触发旧三轮请求数断言均保留原失败状态。smoke的CPU补充只验证其74次真实响应和接入范围，未代替完整三轮审计。性能和质量均按实际阶段界定，未借用历史Math给新版本背书。

原checkpoint SHA256 `14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c`。每天北京时间06:00–24:00可用，当前流水线硬停止2026-10-22 23:58。

新C8有1题达到官方32768-token上限，已计入错误数，最终101/105不排除该题。原审计失败于“INT8预填充全42层”覆盖断言；19种实际M均不在白名单。V2独立复核保留完整旧attention/KV/scheduler/sampling/43bank/M8 NN/M64capture/LM8 BF16和真实BF16回退检查，模型877源不变。后续C64仍要求真正的新M4096 INT8全42层覆盖及>=100/105，未降低这项门槛。
