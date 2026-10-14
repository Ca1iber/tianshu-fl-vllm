# 天数 BI-V150 32g：只改B64为八分段的小规模服务收益3.55%

基于当前正式最好KV16/w1配置，只把B64的adaptive分段4→8，其他批次保持原策略。全局预算2048、原生Prefill M256/KV128/w16/s1/L0、guarded Decode KV16/w1及B24 KV32/w2回退、TopP16、B75G1均不变，地址宽度不变。控制与候选都启用guarded KV16/w1，而不是旧KV32/warp2控制；候选额外加载一份只改choose_segments的进程hook。

新鲜同期双方各三轮4K输入/1K输出、C64/N64，贪心temperature0/top_p1/repetition_penalty1/noLogprobs/ignore_eos/stream，预排首轮。计分均值 **3926.253716→4065.791828 total tok/s，+3.553976%**。这是小规模服务分数，不能作为正式N256记录。

| 组别 | 轮次 | Total tok/s | Output tok/s | TTFT ms | 秒 |
|---|---:|---:|---:|---:|---:|
| control | 1 | 1391.7351 | 278.3470 | 13433.903 | 235.4471 |
| control | 2 | 3926.5198 | 785.3040 | 13060.562 | 83.4530 |
| control | 3 | 3925.9876 | 785.1975 | 13057.418 | 83.4643 |
| candidate | 1 | 1406.6862 | 281.3372 | 13301.304 | 232.9446 |
| candidate | 2 | 4067.2663 | 813.4533 | 13033.064 | 80.5652 |
| candidate | 3 | 4064.3173 | 812.8635 | 13053.569 | 80.6236 |

control计分两轮极差/均值 0.013555%。
candidate计分两轮极差/均值 0.072532%。

384/384请求的实际usage4096/1024/5120、提示种子、文本、TTFT与吞吐已本地逐项核验。192对响应中167对文本相同，7个greedy canary相同，不声称全部输出一致，不替代官方随机采样新Math。7份冻结源、双方12份安装源快照、完全相同server argv、双方75点完整预热和int32 gather核验通过。控制实际B64 S4、候选B64 S8，二者均KV16/w1/s1、L0生成源4a1f8bdf，producer/reducer tile匹配；B24保持KV32/w2。候选策略记录的hook SHA与active_hook及冻结源一致。

原始包3,329,556bytes/13分片及全包SHA通过：`a726704ae702b22a0be94bfdcdeb996989b9c3a90daf20dffe3d5f93a2c8dc94`。原件工作区service-results/20261014_tianshu/kv16_segments_B64S8_service/20261014_kv16_segments_B64S8_service_tianshu。

复核：

```bash
python3 experiments/tianshu_kv16_segments_20261014/verify_downloaded_service.py --root /path/to/20261014_kv16_segments_B64S8_service_tianshu
```

结论：收益达到独立五轮确认门槛，五轮新控制和候选已启动，预排1/2轮；本检查点没有五轮结论，没有新Math或正式分数。正式最好仍3701.31 total tok/s、Math102/105、tag tianshu-best-decode-kv16-w1-20261014，不将旧Math应用到新分段候选。后续准备了有独立fresh标记和5份源码指纹的Math脚本，尚未GPU执行；官方Math C8不代表B64问答覆盖，数值微测与真实4K模型绑定证据分别保留。没有16K服务。
