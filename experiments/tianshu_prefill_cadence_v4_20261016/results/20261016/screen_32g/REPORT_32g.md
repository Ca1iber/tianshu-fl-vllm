# 有界prefill调度：4K/C64/N64配对完成

原serve参数与当前最佳算子下，候选后两轮均值4076.499、同次fresh control均值4038.109 total tok/s，+0.9507%。此为小规模筛选，不是N256官方成绩，未采用，无新Math。前置固定candidate三轮再fresh control三轮，各排首轮，两个arm同一轮输入种子完全相同、轮间不同。

|arm|轮次|total tok/s|TTFT ms|新增调优|采用统计|
|---|---:|---:|---:|---:|---|
|candidate|1|3910.400|15306.176|1|排除|
|candidate|2|4073.979|11958.864|1|是|
|candidate|3|4079.018|11960.214|0|是|
|control|1|3868.213|16150.437|0|排除|
|control|2|4035.081|12761.204|0|是|
|control|3|4041.137|12705.140|0|是|

稳态平均TTFT 12733.172→11959.539 ms（约-6.08%），平均单请求wall 78708.637→79301.320 ms（约+0.75%）。客户端(wall−首个非空文本事件)/1023为64.492→65.828 ms（约+2.07%），不是GPU逐token间隔；文本SSE事件可能跨多个token。吞吐由整组最后完成时间计算，平均请求wall反而增加，不能排除单cohort尾部对齐导致N64收益。下一步单独C64/N128持续补请求验证。

策略为有prefill压力且free block充分时，每三步最多连续两步使纯1-token decode暂时不eligible，原schedule仍负责队列与KV分配。finally恢复eligibility，尊重调用者throttle_prefills及已有未来eligibility/合法修改。只支持本次MiniCPM、WorkerFL、单卡TP/PP/DP/CP1、async、无spec/LoRA/connector/Mamba路径；不承诺请求绝对时延上界。实际configured use_v2=True而WorkerFL使用ModelRunnerFL；候选临时选择原MRV1 cached all_token_ids恢复分支，调用后还原flag。因此候选组合包含恢复metadata兼容路径，不能将全部差额单独归因cadence。

实际候选canary中10个held调度步、5个history payload步；21次恢复请求GPU token与position和调度历史逐项相等。canary同步检查在吞吐前移除arm，之后每轮journal SHA保持不变。原始整数检查行、预期/实际值已下载重验，未声称保存所有GPU Tensor。两组32个并发greedy canary文本跨arm8/16配对相同，仅诊断；不替代Math质量门槛。四个C1 seeded canary相同，384个性能响应全部成功。

候选新增调优在R1/R2各一次_index_linearized_jit_function（不是GEMM），真实benchmark输入长度1323/1345、缓存key1344/1376、int32源/int64索引/int32输出，日志0.19/0.20秒。R3没有新增。不从计时分数减掉这部分成本。75点GEMM预热已经使用，但不意味着所有index shape均被预热。

本地独立核验两arm相同70份冻结源、实际原serve argv、default2048/256seq/75warm/40环境、KV16/S8/.cv、strictTopP32Kgeo、三M2048 launch；全部384响应完整文本/usage/种子/SSE偏移及吞吐重算、16份metrics零抢占通过。当前正式最佳仍4093.81 tok/s、Math102/105。

原始归档13,295,705字节、51分片，整包SHA256 08e84f379dab77b9b3826d770830fb27bae742322e52fd95c89dfd7d21f452d6；原件在service-results/20261016_tianshu/prefill_cadence_v4_service/20261016_prefill_cadence_v4_service_tianshu。
