# v01：三项 sweep 优化后的 4K/C64 阶段耗时复查（C500 64g）

## 1. 上版本遗留问题

旧诊断（reports/15-post-int32-trace/v01_4k_c64_64g）稳态 Decode 中，radix sweep 占 GPU kernel 累计时间 39.14%，Attention 25.25%，原 GEMM 18.46%。
随后分别加入空桶跳过、双桶共享局部前缀扫描、重复 digit 身份排序轮次。需要重新测量真实服务路径的热点，不能继续按旧占比安排优化。

## 2. 问题原因分析

已验证：最新排序在服务中实际执行 _sweep_two_bin_rank；稳态窗口 366 个完整 sort 组，每组 8 次 sweep。
第 1/2/3 pass 明显短于第 4/5 pass，符合重复 digit 检测后身份复制的设计预期。
当前只测耗时，没有新增硬件计数器，因此不能据此认定 Attention 是计算、带宽或某种 stall 瓶颈。

## 3. 本版本解决方案

没有改生产算子。用 mcTracer attach 当前 EngineCore，采一批完整的 4K 输入、64 并发请求。
先用不同随机 token-ID prompt 完成一次 64 并发、2 token 输出的预热，随后诊断输出 512 token，ignore_eos=True，所有请求完整完成。
这是短诊断，不是赛事完整 benchmark；输出 512 不等于官方 1024，不能用此批的整体阶段比例代替正式工作负载比例。

## 4. 具体落地策略

- 机器：968d87327673，C500 64 GB，sGPU Disabled，MACA 3.7.0.38，KMD 3.8.30，mx-smi 2.3.1。
- 日期：2026-10-14；采集约 10:43:15 至 10:43:45（容器日志时间）。
- FlagGems：b765ed231；插件：9aba176。已验证安装 sort.py 与工作区字节一致，SHA 见 installed_sort_sha256.txt。
- 官方 serve 参数不变：模型 /workspace/MiniCPM5-2B，端口 9031，served-model-name minicpm，gpu-memory-utilization 0.85，max-model-len 131072，VLLM_PLUGINS=fl。
- 既有 packed bundle、variable M、75 点捕获保持开启。没有修改官方 benchmark 脚本，没有开启量化或投机解码。
- 诊断：DIAG_MODE=decode PYTHONDONTWRITEBYTECODE=1 python3 -u full_burst.py。
- 解析：PYTHONDONTWRITEBYTECODE=1 python3 -u analyze_burst.py。
- 所有 64 请求 HTTP 200，输入合计 262144 token，输出合计 32768 token；每请求 4096/512，由响应 usage 验证。
- prefix 查询 token 增量 262144，命中增量 0：诊断未使用前缀缓存捷径。
- 原始 trace 565658230 bytes（约 539.45 MiB），文件 full_burst/full_burst-1412229.json；采用流式 JSON 解析，不整体载入原文件。
- 仅统计 ph=X、args.grid 的设备 kernel 事件；类别 cat=0、pid=2。单位 ns，依据当前 mc_mctx_sum.py 的 Total Time(ns)。
- trace SHA 见 raw_trace_sha256.txt；原始大文件保留容器，不纳入 Git。

### 窗口定义

prefill_mixed：请求 burst 到所有 64 客户端观察到首段文本，14.555715 秒。早到的请求已在 Decode，因此不是纯 Prefill。
steady_decode_c64：所有客户端已经生成首段文本、且尚无请求完成，前后各避开 250 ms，10.245839 秒。
decode_tail：首个请求完成到所有请求完成，2.592675 秒，并发会下降。
完整 burst 墙钟 27.894229 秒；稳态窗口外的两个 250 ms 保护区不强行归为稳态。

以上是按客户端响应定义的时间窗口，而不是逐个 token 的 scheduler 标签。
特别不能把 14.555715 / 27.894229 = 52.18% 写成“纯 Prefill 占比”：其中含早到请求的 Decode，本次输出长度也与官方不同。

## 5. Benchmark 对比

本轮未执行新的官方 benchmark，不把 trace 延迟当正式成绩。
第三项优化上一轮已完成的无 trace 官方流程：4K 8248.14 total tok/s，16K 9142.46 total tok/s，出处 reports/16-sweep-empty-bin/v06_repeat_service_64g。
本轮的任务仅重新定位热点；如果后续改 Attention 或 GEMM，再用相同官方参数和完整脚本验证。

## 6. Profile 指标变化

下表分母均为对应窗口的 GPU kernel dur 累计和，不是请求端到端墙钟。

### Prefill-focused / 混合窗口

| 环节 | 调用数 | 累计 ms | kernel 占比 |
|---|---:|---:|---:|
| gate/up + SiLU 融合 | 5460 | 3815.338 | 29.215% |
| down projection | 5460 | 2959.644 | 22.662% |
| Attention（含 combine） | 16253 | 2765.298 | 21.174% |
| QKV projection | 5460 | 1046.230 | 8.011% |
| o_proj | 5460 | 813.923 | 6.232% |
| RMSNorm / residual | 11091 | 432.594 | 3.312% |
| radix sweep | 1040 | 353.522 | 2.707% |

融合 MLP 的 gate/up + down 合计 51.877%。四类 packed 投影分别出现 5460 = 130 × 42 次；这说明采到了 130 组大 M 的 42 层模型执行，不代表“130 个请求”或“纯 Prefill 130 step”。

### 稳态 Decode（64 并发）

| 环节 | 调用数 | 累计 ms | kernel 占比 |
|---|---:|---:|---:|
| Attention（含 combine） | 30751 | 3417.818 | 34.446% |
| 原 GEMM（主干投影合计） | 61501 | 2431.156 | 24.502% |
| radix sweep | 2928 | 1884.268 | 18.990% |
| scatter 恢复 token 顺序 | 366 | 637.519 | 6.425% |
| histogram | 366 | 519.553 | 5.236% |
| packed LM head | 366 | 170.213 | 1.715% |
| RMSNorm / residual | 31116 | 165.050 | 1.663% |
| 两次 softmax | 732 | 70.422 | 0.710% |
| KV-cache 写入 | 15376 | 69.334 | 0.699% |
| 未融合激活 | 15375 | 66.477 | 0.670% |
| 概率/元数据 cumsum | 1098 | 31.837 | 0.321% |
| 桶 prefix | 366 | 1.694 | 0.017% |

其他未单列项仍保留在 family_times_64g.csv，不将其删除或重分配以扩大某项占比。
窗口前/后半 Attention 34.03% / 34.86%，GEMM 24.66% / 24.35%，sweep 19.12% / 18.86%，热点排序稳定。

| 历史稳态诊断对照 | 旧诊断 | 当前 | 占比差（百分点） |
|---|---:|---:|---:|
| radix sweep | 39.14% | 18.99% | -20.15 |
| Attention | 25.25% | 34.45% | +9.20 |
| 原 GEMM | 18.46% | 24.50% | +6.04 |

历史窗口长度、context 进展与采集方式并不完全相同，没有新采旧代码 A/B。占比变化支持热点迁移，不是严格因果归因，也不说明 Attention 本身变慢。

### 每次排序的八个 pass

366 个完整 sort 组中的均值（ms）：P0 0.497602，P1 0.105669，P2 0.105341，P3 0.105520，P4 1.385597，P5 1.383609，P6 1.067740，P7 0.497196；八轮合计 5.148274 ms/sort。
其中 P4 + P5 + P6 约占 sweep 自身 74.53%，比继续消除 P1/P2/P3 的微小成本更值得研究。
scatter 约 1.741854 ms/调用，histogram 约 1.419544 ms/调用；这些不是请求整体 step 的墙钟延迟。

### GPU 活跃区间并集与墙钟

| 窗口 | 墙钟 ms | kernel dur 和 ms | kernel 区间并集 ms | 并集/墙钟 |
|---|---:|---:|---:|---:|
| 完整 burst | 27894.229 | 25725.346 | 25606.631 | 91.799% |
| Prefill/mixed | 14555.715 | 13059.735 | 12941.535 | 88.91% |
| 稳态 Decode | 10245.839 | 9922.367 | 9922.001 | 96.84% |

kernel dur 可重叠，故报告同时记录并集。墙钟剩余部分可能包括 CPU/runtime 间隙、非 kernel 设备事件和采集开销；不能直接判定为 CPU 瓶颈。
没有采 mcProfiler 的 HBM/L2/occupancy/stall，本轮对此均为 TODO，不使用 FP16 MMA 屋顶分析整数排序。

## 7. 实验总结与下一步

稳态 Decode 的第一热点已变为 Attention，第二是模型 GEMM，sweep 为第三；前三项合计约 77.94% 的 kernel 累计时间。
如果继续全局优化，先为本次真实 4K/C64 的 flash_fwd_splitkv_kernel 建独立复现，确认 KV 长度、split 数、tile 与 combine 代价，再测硬件效率。
若继续围绕排序开发，优先排查 P4/P5/P6 或 scatter，而非再压 P1/P2/P3；这需要新的独立测试和正确性验证，不在本次诊断中直接改代码。
Prefill-focused 仍以融合 MLP 为重头，但目前不能把混合窗口说成纯 Prefill，也不能将短诊断输出 512 的整体阶段比例推广到官方 1024。
本轮诊断服务、mcTracer 与分析任务都已结束；不关闭云实例，不创建源码 commit，不 push。

原始数据入口：stage_summary_64g.json、family_times_64g.csv、kernel_times_64g.csv、radix_pass_times_64g.csv、full_burst.observations.json、metrics_before/after.txt、stage_analysis_64g.log。
