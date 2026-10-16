# 天数原始页表H2D与slot mapping成本

使用真实安装BlockTable/CpuGpuBuffer与原方法，page16、maxreq256、maxbatched2048、PCP/DCP1、pinned CPU。8192列由131072/page16源链推导，是控制实例的geometry，并非新完整模型观测；320列仅复制成本参考，未改服务上下文。FlagGems record/once开启。7组交错，每组20次，分别计时commit、slot mapping和二者合计。

|列|活跃请求|上传bytes|commit GPU ms|slot GPU ms|合计GPU ms|合计host wall ms|
|---:|---:|---:|---:|---:|---:|---:|
|8192|1|32768|0.003487|0.004769|0.008145|0.074967|
|8192|8|262144|0.012788|0.005716|0.021310|0.085307|
|8192|64|2097152|0.089324|0.006638|0.090543|0.156809|
|320|1|1280|0.002131|0.004760|0.006823|0.072595|
|320|8|10240|0.002477|0.005727|0.008095|0.074030|
|320|64|81920|0.005008|0.006640|0.011558|0.078025|

6场景、12次改变CPU页号与GPU positions、清空输出后Graph重放，完整活跃行/padding、真实slot ID和尾部-1均检查通过。保留int64 positions/slots。检查由GPU运行时完成，raw没有完整Tensor序列化，本地只独立核对布尔记录、126个原始计时、重算中位数以及4份源码快照与当前仓库probe字节。

64请求的2 MiB整行上传约0.089 ms GPU，含slot mapping约0.091 ms。现有成本很小，决定不引入复杂增量同步。潜在增量实现还需正确覆盖add/append/move/swap/clear、直接CPU别名写入、GPU初始化/padding与异步staging复用；不能只看字节数就改。不同计时项不可相加推断CPU/GPU重叠或模型百分比。

没有候选服务、Math或正式benchmark，未采用；正式最好仍4093.81 total tok/s、Math102/105。原始归档3949218 bytes、16分片，SHA256 e5375d90c48564d8dd36dc0aa728a0eb817f308a630834333e44214ea71bf6db，全分片与整包verified。raw见service-results/20261016_tianshu/block_table_cost/20261016_block_table_cost_tianshu。
