# 天数 BI-V150 32g：禁用异步调度为负优化

本轮新鲜3+3服务对照，4K输入/1K输出、C64/N64、每组预排除首轮。两组均CV/KV16/w1/S8、budget4096/83预热点、maxseq256、原生Prefill/TopP、BF16、temperature1/top_p.95。候选只追加--no-async-scheduling；真实Worker配置控制True、候选False。

|组别|轮次|total tok/s|output tok/s|TTFT ms|
|---|---:|---:|---:|---:|
|异步开启|1（排除）|3806.02|761.20|17431.55|
|异步开启|2|4036.04|807.21|11779.99|
|异步开启|3|4053.36|810.67|11773.11|
|异步关闭|1（排除）|2876.26|575.25|35600.82|
|异步关闭|2|3800.45|760.09|11958.30|
|异步关闭|3|3781.06|756.21|12179.24|

稳态4044.6991→3790.7539 total tok/s，下降 **6.278470%**；稳态平均TTFT约高2.48%。保持异步开启，不采用，不新跑Math或完整benchmark。不能把该N64短测与正式N256最佳4091.29混为同条件成绩。

384条吞吐响应全部成功，并本地复核输入生成种子、用量4096/1024、正文非空、逐请求TTFT/墙钟与聚合吞吐。17份冻结源码及快照、6安装源、83实际预热、CV生成源码及KV16/S8 producer/reducer、真实Worker async布尔值、maxseq256、两组命令仅异步开关不同全部核验通过。两组每轮记录的autotune均为0。未采集新硬件profile，不将差异归因到某个未经测量的内部步骤。

归档5322317bytes/21parts，SHA256 `f1f9a0ff5929c575774f92eb61964564ed034dc027a5f86e5682cb5eda89f61a`。本机raw：`service-results/20261016_tianshu/async_schedule_service/20261016_async_schedule_service_tianshu`。

复核：`python3 experiments/tianshu_async_schedule_20261016/verify_downloaded.py --root <raw-root>`。正式最佳4091.29/Math103of105保持。该实验已终态并释放GPU，后续五项小收益组合的独立3+3筛选已串行启动。
