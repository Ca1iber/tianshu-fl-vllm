# 天数有界prefill调度V3：直接检查恢复输入

V1的MRV1-only守卫与实际配置不符，已中止保留。V2补齐元数据后，未启用策略的control两次greedy并发探针已有4/8全文不同，0轮吞吐、candidate未运行；不能把这种对照本身不满足的逐字门槛当作新策略正确性。原因未据此归结为某个硬件或数值缺陷，旧失败记录不改成通过。

V3的policy与V2字节相同：WorkerFL TP/PP/DP/CP1等守卫，原schedule调用期间选已有MRV1 token历史恢复路径，finally恢复配置flag，原2048预算且每三步最多连续推迟两步decode。控制仍原调度与配置flag，候选是元数据兼容+节奏包。32条greedy并发探针仍完整保存并比较，全文相同数量只是诊断，不当质量结论。

新增精确检查：仅并发canary阶段，对未在前一批次中、已产生输出、当前恰好调度1 token且带all_token_ids恢复历史的请求，在真实_prepare_inputs返回后读取GPU input_ids和int64 positions，逐项检查等于调度器canonical history[num_computed_tokens]和位置。候选必须至少检查一个实际恢复请求且实际发生deferral/history payload。GPU同步检查只在显式arm存在时执行，吞吐前移除arm，每轮后检查journal SHA保持不变。CPU正例、错误token/位置/缺失history负例以及未armed不访问GPU的合同通过；这不是完整Scheduler模拟或模型准确率。

随后原serve命令、当前最佳CV/KV16/S8/GEO/M2048算子，4K/1K C64/N64 3+3，预排首轮。记录全部响应、TTFT、客户端decode均摊间隔与文本事件间隔。只有明确收益才新Math105>95和正式N256测试。最好4093.81/Math102不适用于未验证的新策略；无16K、安装/权重/原benchmark/最佳启动器/沐曦修改，23:58 GPU硬停。
