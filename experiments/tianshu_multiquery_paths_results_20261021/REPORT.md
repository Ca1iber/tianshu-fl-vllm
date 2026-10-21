# 天数多查询 attention：原生 gather 路径与 Split-KV 对照

26 场景全部执行并完成独立 CPU 复核；这是 attention 读取组件测试，尚未接入模型，未替换推荐版。普通非投机 4K 的实际 prefill 最小查询长度为 128，不能把 Q≤8 组件结果算作普通 4K 收益。

原始数据：`/workspace/benchmark-results/20261021_multiquery_paths_GPU_tianshu`。原始结果 SHA256：`7f2996b0626b7a54dbd727a35b83bd581c98404bb08d297f6533cfecf3b325b4`。25 项冻结源闭包未修改。全部结果见 [CPU 校验](../tianshu_multiquery_paths_recovery_20261021/verified/GPU_GATE_32g.json) 和 [完整计时](../tianshu_multiquery_paths_recovery_20261021/verified/KEY_TIMINGS_32g.csv)。

| 场景 | 当前路径 ms | 原 Split-KV Q4/T32/w2 ms | 本轮最佳 ms |
|---|---:|---:|---:|
| N1 / Q3 / K4096 | 0.290715 | 0.033764 | 0.033764（原 Split-KV） |
| N8 / Q3 / K4096 | 0.454822 | 0.129710 | 0.129710（原 Split-KV） |
| N32 / Q3 / K4096 | 2.108567 | 0.451679 | 0.451679（原 Split-KV） |
| N64 / Q3 / K4096 | 23.971686 | 0.892061 | 0.892061（原 Split-KV） |
| N64 / Q4 / K4096 | 23.937984 | 1.727730 | 1.169128（Q2/T16/w1） |
| N64 / Q8 / K4096 | 23.867654 | 1.792874 | 1.792874（原 Split-KV） |
| N8 / Q8 / K8 | 0.022088 | 0.028404 | 0.025236（Q2/T16/w1，仍慢于当前路径） |

当前路径在容量允许时包含实际 `SmallArena` + bounded gather + 原生 BF16 attention；超容量使用当前 byte 2D reader。表中均匀、因果查询对应当前目标模型路径。容量内混合查询的 gather-all 只是算术参照，生产路径会拆分 decode；非因果场景也未声称已经接入草稿模型。四种候选为探索配置，非独立留出验证。

每场景两次改变输入/KV/长度/页表的完整输出校验，随后七轮轮换顺序、每轮 20 次 CUDA Graph replay。共 260 个完整输出、910 个计时样本。CPU 独立复核 1,096 个理想 attention 向量、117,988,352 个 K codec 数值、50 个完整短 K 原生 BF16-P 输出。长 K 使用原有 FP32 误差阈值；短 K 对照原生 P 转 BF16 后的在线 softmax/PV 语义，同时保留理想 FP32 阈值下的差异，未提高原阈值。

逐 phase 验证输出、scratch、KV、输入、metadata、RNG。原生 arena/ID guard 实际记录范围为每 case 全部计时结束后，26 项全通过；不能声称逐 phase 检查该 guard。冻结 CPU 归档器曾错误索取不存在的逐 phase 字段，因此采用独立 CPU 恢复器，未改原源、原 GPU 结果，未重跑 GPU，详见 [恢复说明](VERIFIER_RECOVERY.md)。

结论：两步草稿对应目标验证 Q3 时，原 Q4/T32/w2 Split-KV 仍优于新 T16 配置。三步草稿对应 Q4 时，新 Q2/T16/w1 相对原 Split-KV 时间减少约 32.3%。短 K 并非普遍提速。后续应在真实投机模型中验证路由命中、47 层 KV 与 scratch 的原生显存计费、草稿接受率及整体吞吐；当前不据此更换推荐版或宣称整模型提升。
