# 当前K8V16/default4096模型阶段测量

沿用已取得5101.45正式4K及Math102/105的205源、43环境项模板。新hook通过已有finder链组合，保留原始框架默认4096、max-seqs256、异步调度、K8V16、严格TopP、原RNG及83点预热。只额外启用4项测量配置、将PYTHONPATH转到测量入口。

复用10月16日通过的Graph外事件测量：连续32步C64预热后记录40步FULL Graph；分别记录forward、词表投影、sample及总stream区间，以及host wall/thread-CPU。测量事件在Graph外，只有窗口边界同步。区间包含host提交空隙，不能当单kernel活跃耗时或硬件occupancy。

预热/测量各64请求，4096输入/512输出；seed777 canary前后必须一致。原始回复及完整运行证据留容器，尚不是原始benchmark成绩。使用当前K8V16 runtime审计器，唯一审计适配为环境项43→47，模型/算子/调度断言不变。
