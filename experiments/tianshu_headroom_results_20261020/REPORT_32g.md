# KV水位1/16：抢占6→0，16K N64吞吐+0.02269%，不采用

2026-10-20，BI-V150 32G，两组均为已验证K8V16/4096/S16配置、原始serve argv。唯一配置差异是原生KV watermark 0→1/16。C64、16K/1K、N64，每组3轮排首轮；种子按轮变化、组间相同。离线核验234份相同源码及快照、48项环境、全部384/384成功、逐请求完整usage/指标、两组seeded canary相同、实际manager水位及真实FL Graph捕获重放通过。本实验是服务筛选，不是官方完整N128 benchmark。

| 模式 | 计分总tok/s | 平均TTFT s | 计分吞吐极差/均值 | 三轮累计抢占 |
|---|---:|---:|---:|---:|
| 原生水位0 | 3748.76125 | 110.28399 | 0.01464% | 6 |
| 水位1/16 | 3749.61191 | 111.55846 | 0.01629% | 0 |

吞吐变化+0.02269%，TTFT略增约1.27秒，没有足够收益支持采用或重跑Math/完整双负载。两组分别42378/42386个实际KV块，候选保留2649块；内存profile的小差异也需保留，不把微小分差作收益。manager诊断确认水位实际生效，原生allocate_slots实现未修改。

实际Graph累计重放显示，控制组主要批次39/40和24/25，候选主要38和26；详见[GRAPH_BATCHES_32g.json](GRAPH_BATCHES_32g.json)。这与较保守准入改变并发和尾批次一致，可以解释为什么消除抢占仍未带来整机改善，但不是单因素硬件因果证明。累计记录包含warmup/canary，不当作逐轮耗时profile。

逐轮CSV见[SERVICE_16K_ROUNDS_32g.csv](SERVICE_16K_ROUNDS_32g.csv)，完整独立核验见[SERVICE_16K_GATE_32g.json](SERVICE_16K_GATE_32g.json)。原始响应、服务日志、实际manager记录与trace留容器`/workspace/benchmark-results/20261020_headroom_service16k_tianshu`。新正式发布仍4K5099.65、16K3723.12 total tok/s与Math102/105，不含本候选。
