# 天数有界prefill V4：候选先行、移除收尾1-token探针

V3控制的192个4K/1K性能请求和3轮全部完成，后两轮4033.396/4039.473 tok/s，无新增调优；失败发生在追加的1-token收尾诊断请求客户端检查，after_metrics尚未取得，所有70源未改。旧failed状态/响应/源码保留，不把它冒充完整正常对照。

V4沿V3完全相同的policy和GPU恢复输入检查器字节，运行中server hook只对应新目录同内容，保留全部真实canary/输入历史/位置/40env/75warm/原2048预算/原serve argv的检查。移除不属于吞吐或质量的收尾1-token探针，之后直接获取metrics并核验源码；异常额外保留完整traceback。

为了先验证实际候选，本轮预先固定执行candidate三轮，再启动全新control三轮；均C64/N64 4K/1K、每组排首轮、seed列表相同。不能把旧V3 control当作此次正式配对来源。框架策略不变：WorkerFL TP/PP/DP/CP1，调用原schedule时选择原MRV1 token恢复元数据，finally还原配置flag，每三步最多连续让出两步decode给prefill；控制原flag/原调度。直接GPU token/position检查仅在canary arm开启，吞吐前关闭、每轮后保持journal SHA不变。greedy文本一致数量仅诊断，候选仍须实际恢复检查通过，清晰增益后需新Math105>95与原N256正式。

不改旧冻结目录、权重/安装源/最佳启动器/原benchmark或沐曦；无16K，23:58 GPU硬停。最佳4093.81/Math102保持。
