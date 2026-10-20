# 当前K8V16/default4096模型阶段测量

沿用已取得5101.45正式4K及Math102/105的205源、43环境项模板。新hook通过已有finder链组合，保留原始框架默认4096、max-seqs256、异步调度、K8V16、严格TopP、原RNG及83点预热。只额外启用4项测量配置、将PYTHONPATH转到测量入口。

复用10月16日通过的Graph外事件测量：连续32步C64预热后记录40步FULL Graph；分别记录forward、词表投影、sample及总stream区间，以及host wall/thread-CPU。测量事件在Graph外，只有窗口边界同步。区间包含host提交空隙，不能当单kernel活跃耗时或硬件occupancy。

预热/测量各64请求，4096输入/512输出；seed777 canary前后必须一致。原始回复及完整运行证据留容器，尚不是原始benchmark成绩。使用当前K8V16 runtime审计器，唯一审计适配为环境项43→47，模型/算子/调度断言不变。

## 已完成结果

40步均为真实C64、FULL Graph、padded64，测量窗口前已有32步稳定解码。每步stream区间中位数：总39.628233ms，forward34.976412ms，词表投影1.134342ms，sample3.382758ms；40步区间累计占比依次88.2247%、2.8621%、8.5589%。执行CPU调用中位数3.969890ms，采样调用1.956904ms；CPU与GPU不能直接相加，异步调用存在重叠。

当前205源基线保留，新增测量及实际compute_logits源后冻结210源。128个负载请求全部完成，前后seed777 canary一致，原始命令/实际47环境/4096预算/83预热/K8V16/原RNG/零preemption通过完整审计。未测新Math或正式benchmark，正式最好仍为5101.45。结论是继续优先研究模型前向；这不是各个kernel活跃时间的排序。

STAGE_GATE_32g.json 保存冻结源及实际配置，STAGE_FRAMES_32g.json 保存40步原始事件，STAGE_SUMMARY_32g.json 保存汇总。完整响应和日志留容器 `/workspace/benchmark-results/20261020_stage_current_tianshu`。
