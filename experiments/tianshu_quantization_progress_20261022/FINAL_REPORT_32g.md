# 天数量化探索最终报告，2026-10-22

本轮找到并交付了一条有效路线：复用既有A8/W8 NN权重布局，用原生INT8 GEMM替代M512/4096的原Triton双dot预填充计算。原模型、架构与checkpoint保持不变，未训练/微调，没有新增另一套静态权重。

新正式4K/1K、C64/N256、四轮排首轮：**5777.22 total / 1155.45 output tok/s，平均TTFT 3459.44 ms**。相对旧正式5695.95吞吐+1.43%，TTFT降低5.58%。同期同源N64 A/B提升2.16%，二者口径分别保留。

|探索|结果|处理|
|---|---|---|
|W4/W8新权重组件|9版、300配置、900变化输入相位独立数值审核；测试目标均慢于现用控制，最快W4A8 Down仍多28.1%耗时|保留负结果，不接模型|
|A8倒数边界修正|有限BF16域量化码/scale审核通过，6个性能点均慢于现量化器|不采用|
|原生INT8 GateUp M64|完整路径0.066165ms，对照0.063122ms|保持旧M64融合|
|原生INT8 M512/4096，全42实际MLP bank含Down|完整组件逆耗时改善6.16%/18.07%，变化输入、整图和独立CPU审核通过|接入完整服务验证|
|真实模型N64同期对照|5600.66→5721.43 total tok/s，+2.1563%，候选热轮差幅0.09224%|进入新质量和正式N256|
|新C8与实际C64 Math500 Level3|各全105题、均101/105=96.1905%；C64实际M4096新INT8全42层|精度门槛通过|
|新正式N256|5777.22 total tok/s，1024请求全成功，热轮差幅0.6353%|晋升4K推荐版|

## 交付与复现

[新推荐版直接Bash入口](../tianshu_native_int8_prefill_release_v2_20261022/README_32g.md)，[正式逐轮数据与范围](../tianshu_native_int8_prefill_official_results_v2_20261022/REPORT_32g.md)，[双并发独立质量](../tianshu_native_int8_prefill_quality_results_20261022/REPORT_32g.md)。代码、报告、关键JSON/CSV保存在天数GitHub；完整数组、cache、profile和原始回答仅留容器，无本地项目副本。

本版877份模型源中原471份精确不变；20项控制器、原13条serve argv、63项选定环境、384份cache文件校验通过。192MiB临时INT32工作区已在原生profile中全42层实际执行并计入KV预算。Bash入口从/root和/tmp做无GPU dry，命令/实际环境/工作目录与正式profile一致；没有再启动GPU服务器。

新C8预填充实际全BF16、INT8命中0，原GPU覆盖审核audit_failed保留；独立CPU按实际覆盖复核。新C64原GPU完整审核complete，新INT8 M4096全42层、M64融合图和LM64 INT8均已实测，全部105题自然stop。正式GPU程序原四轮benchmark全成功，收尾因缺workload metadata为audit_failed；独立CPU只由原命令和4轮CSV恢复审核上下文，保留全部原检查并实际验证零抢占，不修改原result/CSV。所有原失败hash和恢复范围均明确写入门槛，不将CPU恢复冒充原GPU程序通过。

原checkpoint最终SHA仍14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c。就绪后Triton autotune日志0、零抢占。16K本版未复测，不引用旧16K成绩，不宣称完整比赛资格。旧5695.95入口及所有失败/负结果保留。第一阶段REPORT_FINAL的收尾属于当时的负结果阶段，本报告纳入后续原生INT8新证据。

阶段代码、计划、数值边界与全部关键原始hash见相应实验目录。W4等负结论只限已测实现/shape/tile；MLP组件18.07%与整机1.43%来自不同范围，不混为同一个性能结果。
