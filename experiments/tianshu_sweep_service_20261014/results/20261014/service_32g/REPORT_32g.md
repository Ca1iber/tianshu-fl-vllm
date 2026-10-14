# 天数BI-V150 32g：小批次sweep排序服务对照

两边均采用当前4096预算/83点、KV16/S8、SQL、NativePrefill及原生TopP配置。4K输入/1K输出、C64、N64、temperature1/top_p.95，每边新启动3轮，预先排除首轮。候选只替换小batch TopP回退中的一次排序，保护B1..7连续CUDA FP32、词表130560；B>=8继续原生TopP，采样掩码与RNG不变。完整版本含原有INT32/prefix优化，不把组合收益归为单项机制。

|组别|第1轮（排除）|第2轮|第3轮|后两轮均值 tok/s|
|---|---:|---:|---:|---:|
|原配置|3738.29|3955.36|3957.36|3956.36|
|小batch sweep|3391.29|3970.07|3961.90|3965.98|

稳态提升仅 **0.2432%**，暂不采用，不因此启动完整benchmark或新Math。此前28%～42%的独立排序耗时下降没有转为同等服务吞吐提升。

实际路由记录：两组完全一致，整个worker生命周期（含启动/预热、canary与3轮测试）各3268次apply_top_k_top_p；小batch回退42次，B1..7各6次，占1.2852%调用次数；候选确实42次调用新排序，control为0。B64为2883次，B>=8总3226次，原生TopP不变。调用次数不是GPU耗时占比，也不是只覆盖计分后两轮；不以计数冒充profile。

全部384吞吐请求用量、种子、文本、TTFT与耗时重算通过；15份冻结源码、实际sampler SHA、候选源码与helper SHA、mmap原始计数、双方83点预热、KV16/S8与producer/reducer匹配、安装6源快照均本地复核通过。没有将随机文本差异当成正确率。

归档4958517bytes/19parts，SHA256 `67f71aa8781a282e6e8be62c4b0046a60573d149da6c71bff18d306fd5535117`。本机raw `service-results/20261014_tianshu/sweep_sampler_service/20261014_sweep_sampler_service_tianshu`。复核命令：`python3 experiments/tianshu_sweep_service_20261014/verify_downloaded.py --root <raw-root>`。

当前正式最佳4000.99 total tok/s、Math103/105保持。排序方案保留，未来不同并发/长度分布或Math尾部可能相关，但本轮不外推收益。下一实际服务实验为Decode K/V加载.cv缓存提示，当前micro已有B64约5%耗时减少证据。
