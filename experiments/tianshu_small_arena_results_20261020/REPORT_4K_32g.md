# 小暂存区4K N64：5031.30→5030.59 total tok/s，基本持平

2026-10-20，BI-V150 32G，原始serve argv、K8V16/S16、4096预算、83预热点、max_num_seqs256与KV水位0相同。唯一配置差异512→128MiB prefill复用arena，两组236份相同源、48项环境，N64 C64各3轮排首轮。独立核验全部384/384请求及usage、组间逐请求输入种子一致/跨轮不同、seeded canary相同、真实native FL Graph和实际reserve调用栈通过。

| 模式 | 首轮（排除） | 第二轮 | 第三轮 | 计分均值 | 计分极差/均值 | 平均TTFT s |
|---|---:|---:|---:|---:|---:|---:|
| 512MiB | 4746.42664 | 5031.81123 | 5030.78826 | 5031.29974 | 0.02033% | 11.59918 |
| 128MiB | 4760.98364 | 5034.49713 | 5026.69098 | 5030.59406 | 0.15517% | 11.55471 |

计分吞吐变化-0.01403%，不作性能收益。两组都0抢占、0错误，实际prefill暂存最大520页，无bounded_fallback，记录中的Python调用计数不是Graph replay次数。实际KV容量678176→690656 tokens，增加12480；未据此直接预测吞吐。16K小规模对照仍在运行，尚未采用、未跑新Math/完整benchmark，当前正式最佳不变。

关键证据[SERVICE_4K_GATE_32g.json](SERVICE_4K_GATE_32g.json)与[SERVICE_4K_ROUNDS_32g.csv](SERVICE_4K_ROUNDS_32g.csv)，完整原件留容器`/workspace/benchmark-results/20261020_small_arena_service4k_tianshu`。服务原始command、模型和benchmark均未修改。
