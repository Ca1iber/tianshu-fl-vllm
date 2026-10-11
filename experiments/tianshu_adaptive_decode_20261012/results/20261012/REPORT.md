# 天数原生 Decode 自适应分段实验

结论：候选通过独立 MATH-500 Level 3：102/105，97.142857%，105请求全部成功、无错误。官方门槛为严格大于95%，已经通过。短服务对照有约3.23%吞吐提升和17.20%单请求耗时下降，可保留为后续框架优化的基础；该候选尚未完成正式256请求4K复测。

控制为已验证 Native Decode + FA2 Prefill（正式4K2286.09 total tok/s、TTFT7569.57ms，MATH100/105）。本轮保持旧 Hybrid 源码不变，新增独立开关 `VLLM_FL_ILUVATAR_ADAPTIVE_DECODE_SEGMENTS`，通用默认0。按Graph静态批次选择：1→64段，2–15→16段，16–64→4段。Builder一次分配64段scratch，native两个kernel使用逻辑packed前缀，避免运行时GPU值同步、临时调优和重复分配。

80配置通过固定FP32 atol0.02/rtol0.01、独立CPU参考以及未用scratch tail检查；13项真实metadata/Hybrid/CUDA Graph联合检查通过。B1/4K约1.88x，B1/32768生成历史约2.90x；B4以16段为佳，64段更慢，所以没有统一增加分段。B16/32/64长历史4段和16段相近。32768仅为生成历史数值/算子测试，没有运行完整16K吞吐benchmark。

| 缩小服务实验，跳首轮 | 控制 | 自适应 | 变化 |
|---|---:|---:|---:|
| 4K输入/1K输出，64并发/64请求，total tok/s | 2322.92 | 2398.00 | +3.2322% |
| 同一实验平均TTFT，ms | 23186.42 | 23326.66 | +0.605% |
| 4K输入/4K输出，单请求耗时，s | 67.8360 | 56.1689 | −17.1990% |
| KV容量，tokens | 521936 | 521936 | 相同 |

每组202请求，共404全部成功。两组各三轮并排除首次；短实验的批次填充/排空比例与官方256请求不同，不作为官方成绩。新增scratch理论约24.4MiB，但未测到KV容量损失。

独立数学使用官方batch8、temperature1/top_p0.95/max_tokens32768与本地Level3数据集。实际计算源冻结，未复用旧数学成绩。105预测及105评分均唯一且成功；错误项18、87、101；104项正常stop，1项max_tokens（101，保留官方错误判定）。102/105通过规定门槛，一次随机采样结果不用于推断精度提升。

实际新helper SHA256为e5d983e43c3b343afb558440a9114255cebc5d01569cc192191290b0bb0c84c8；Hybrid SHA为5c55a8d57e438b36d176555fd9273ab9ba665dada4c57d57c19633394c2ed4e4；Worker SHA为6cf62f60ca8ea12b7c494f0009fa733f33270a73e014cdbd4a45cada787c032a。实际路径、原始响应、日志、全部计时及数值结果位于远端/workspace/benchmark-results/20261012_adaptive_decode_tianshu及本地service-results/20261012_adaptive_decode_tianshu。

数学完整备份已验证SHA256：3cb3426c5964a3ed33a3b149bbc3f75f49cd0e8d948b8361705a6e0ec0d7c401，901125 bytes，原始评分/预测已提取到math_adaptive。此前短服务完整备份SHA为75dc6a9aaf999a71cbc81d9d091367bc6360fa82902606040bb5ae5b264f930f。所有GPU任务串行，仅清理自有服务。推荐profile保持官方模型与启动参数；后续只在独立实验中调整Prefill预算。
