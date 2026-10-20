# GateUp NN：组件收益未转成明确模型收益

2026-10-21，BI-V150 32G。两组均原始 13 个 serve 参数、381 运行源码 / 62 环境、4096 预算 / 64 序列 / 异步调度；模型量化范围和逻辑权重相同。候选仅 42 个 GateUp M8/M64 使用 NN 实体布局与组件选定分块，LM8 BF16 / LM64 NT INT8、BF16 prefill 和 attention 路由保持原方案。完整逻辑 bank SHA、42 层 M8/M64 实际捕获路径、内核/配置/二进制记录及原生调用审计通过。

| 组 | 轮 | total tok/s | output tok/s | TTFT ms | 新调优 |
|---|---:|---:|---:|---:|---:|
| control | 1，排除 | 5232.45 | 1046.49 | 12932.21 | 0 |
| control | 2 | 5430.86 | 1086.17 | 10904.44 | 0 |
| control | 3 | 5412.88 | 1082.58 | 10932.27 | 0 |
| candidate | 1，排除 | 5243.97 | 1048.79 | 12903.02 | 0 |
| candidate | 2 | 5440.07 | 1088.01 | 10865.13 | 0 |
| candidate | 3 | 5419.10 | 1083.82 | 10913.22 | 0 |

计分均值 5421.869811 → 5429.587101 total tok/s，+0.142336%；TTFT 10918.358194 → 10889.179483 ms。组内吞吐差 0.331679% / 0.386306%，差值不足以建立稳定端到端收益，不采用、不升级 Math / 正式 N256，不跑新 16K。

384 个计时请求全部成功，seed canary 跨组相同，零抢占，FutureKV 全释放。两组 GateUp 权重 payload 同为 1,056,964,608 字节，43-bank 总 payload 仍为 1,326,938,112 字节；真实 KV block 40723 / 40658。字节数不变不等于运行时 KV 容量必然完全相同，本轮没有新增内存 profiler 归因，不能断言差异原因。

旧 dense binding 的 kernel/config 字段描述原 NT 模块；实际 GateUp NN 执行以新增 NN_gateup_record 的内核 SHA、分块、编译资源和捕获图为准。Python 调用计数不当作 Graph 重放次数。组件 M8/M64 的 7% 局部结果不能当模型加速比例。

关键证据 SCREEN_GATE_32g.json / SCREEN_ROUNDS_32g.json / KEY_ROUNDS_32g.csv。原始日志和完整审计仅容器 /workspace/benchmark-results/20261021_int8_nn_gateup_service_{control,candidate}_tianshu。当前正式推荐 5482.79 total tok/s、Math 101/105，以及吞吐替代 5509.23 / Math 102/105 都未被本次替换；N64 与 N256 不作无对照直接比较。
