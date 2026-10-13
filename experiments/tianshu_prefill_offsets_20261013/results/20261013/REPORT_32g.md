# 天数 BI-V150 32g：Prefill stage2统一配置验证

同一配置正式4K总吞吐3213.13 tok/s、TTFT4456.78ms；正式16K总吞吐2665.42 tok/s、TTFT184434.62ms；独立Math Level3为103/105（98.095238%），0执行错误。全部1536个正式benchmark请求成功。两种负载均使用未改动的官方脚本，各四轮排除首轮；三个测试的stage2/Decode32源SHA和实际Prefill配置核验一致。

|负载|上一套共同配置|新共同配置|吞吐变化|官方公开基线|相对公开基线|
|---|---:|---:|---:|---:|---:|
|4K/1K C64 N256|3202.05|3213.13|+0.35%|2028.01|+58.44%|
|16K/1K C64 N128|2600.48|2665.42|+2.50%|915.15|+191.25%|

历史4K最高仍是3222.52（旧Prefill stage1、无Decode32配置），新共同配置低0.29%；保留旧4K参照及其可回退tag。不能将旧3222.52和新2665.42标成同一配置成绩。4K后三轮波动3.76%，16K0.257%；4K的0.35%增量不构成明确收益，继续优化重点放在4K。

共同配置为Hybrid、native GQA Prefill M256/KV128/w16/stage2、无sink时L0初始化、adaptive Decode KV16/w4/stage1、B4..48满足边界检查时使用32位Decode元素偏移、budget2048、75点Prefill预热桶与固定末token gather。BF16、不投机、不量化。此次增量只改变Prefill的num_stages 1→2，核函数body和网格不变，Prefill偏移仍为64位。

验证由96个CPU FP32检查和GPU参考、位级/尾部/CUDA Graph检查开始，再完成同期服务A/B，最后独立Math和正式负载。正常16K/1K N64小测试+2.43%，prefill偏重的16K/128 N32小测试+5.39%；这些小测试不替代正式计分。扩大M/KV tile的部分方案超过编译器报告的共享内存限额而失败，未采用。Prefill偏移32位仅0.17%~0.44%微测收益，未采用。硬件性能剖析未采集，TODO。

|归档|bytes|SHA256|
|---|---:|---|
|独立Math|3081325|dce2352578f3ba6aee04e41bff9275e45780673ed217d6f343df770dd367e80e|
|正式16K|2348350|0b35cf62b7315f49e7eca41bee9a666dfce0aecbcfa55e96c336b4c17b1d1d45|
|正式4K|2383108|c14929b807c85ca65fc308499afc0806303a96245f848bc684d9202d04913899|

三个归档完整下载并核验分片和整包，原始响应/CSV/实际配置记录/源快照存于本地service-results/20261013_tianshu/prefill_stage2_*，容器仍有原件。匹配启动脚本为experiments/tianshu_prefill_offsets_20261013/serve_tianshu_minicpm.sh。

今后默认用4K/1K C64 N64的同期对照筛选有效方向，有明确提升再跑原始4K N256；每轮向用户报成绩。16K仅在用户明确安排时再测。旧continue_stage2_validation.py保留本次完整执行的历史源码，其16K自动顺序不再用于后续优化。
