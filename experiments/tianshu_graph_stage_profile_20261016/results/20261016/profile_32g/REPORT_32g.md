# 天数 BI-V150 32G：当前原始命令模型阶段诊断

2026-10-16。本轮只定位耗时，未产生新优化成绩。正式 4K 仍为 3992.97 total tok/s、798.60 output tok/s、TTFT 4443.93 ms；质量为 103/105（98.095%）。保留 release `tianshu-original-command-4k-cv-20261016`，没有采用上一轮仅 +0.36794% 的 RNG 采样融合，也没有再跑 Math 或正式 benchmark。

## 结论及使用边界

当前稳定 C64 decode 的模型前向流区间约占 88.48%，应优先继续拆查前向内的 GEMM、attention 等实际路径。仅凭本轮不能判断其中哪个算子最大。词表投影约 2.10%，完整采样约 9.16%；上一轮 RNG 融合只缩短采样的一部分，不能将其单算子约 -53% 当成整个采样或模型吞吐的收益。

这些是 GPU 流中两枚图外 event 之间的时间，可能包含提交空隙及等待；它们不是逐 kernel 活动时间。加 event 和 Python 包裹自身也有成本，本轮没有测量该成本。CPU 调用跨度包含等待，线程 CPU 时间仅涵盖被测工作线程；GPU 与 CPU 时间存在重叠，不能相加。未将这些诊断数据换算为比赛吞吐。

## 真实模型结果

源提交 `2f7a03bf4b373837d000a2ecc909a17d02f312a5`。沿用官方原始 `vllm serve` argv；实际 Worker 默认 budget=2048、max_num_seqs=256、async=True、stream_interval=1。75 个实际预热点已核验，保留 CV/KV16/S8、原始 attention int64 偏移、last-token gather int32、nativeTopP 和原采样器。39 个实际环境项包含 4 个显式诊断标志。

先完整执行 N64/C64、4096 输入/512 输出预热，再换一组输入种子执行相同负载。两组分别用时 57.032811 s 和 53.188812 s；这是包含 prefill 的诊断负载时长，不是 4K/1K 官方 benchmark。每组 64 条响应均保留正文、usage、TTFT 和墙钟耗时；输入种子不同，避免重复输入命中前缀缓存的误导。

arm 后先等 32 步连续的 C64 单 token decode，随后记录 40 步。每步实际 FULL Graph/padded_tokens=64，所有 prompt=4096；首步 computed 范围 4128..4256，末步 4167..4295，各步递增 1。事件预初始化完成后才进入窗口，窗口内不逐步同步，两端各同步一次。`compute_logits` 实际为 `LlamaForCausalLM.compute_logits`，其安装源码另行快照，故最终冻结源码共 50 份。

| GPU 流区间 | 每步中位 ms | 40 步累计 ms | 占总累计 |
|---|---:|---:|---:|
| 整步 | 54.179207 | 2165.712357 | 100% |
| 模型前向 | 47.932198 | 1916.289757 | 88.4831% |
| 词表投影 | 1.138156 | 45.444416 | 2.0984% |
| 采样 | 5.033775 | 198.362650 | 9.1592% |
| 其余区间 | — | 5.615534 | 0.2593% |

占比用累计量计算；各区间中位数不能直接相加为整步中位数。

| 主机调用 | 墙钟中位 ms | 线程 CPU 中位 ms |
|---|---:|---:|
| execute_model | 4.036160 | 4.025833 |
| sample_tokens | 1.966955 | 1.967104 |
| model forward 调用 | 0.064907 | 0.064824 |
| compute_logits 调用 | 0.239554 | 0.240096 |
| _sample 调用 | 1.354111 | 1.354805 |

forward 主机调用只负责提交 Graph，不能拿 0.065 ms 代表模型执行耗时。小于微秒量级的 CPU/墙钟差异不作等待分析。

前后固定 API seed777 的 128 输入/16 输出 canary 文本完全相同。128 条负载请求均有 4096/512/4608 usage；无 preemption。服务退出后 `ixsmi` 确认无 GPU 进程。

## IXsys 能力检查：保留失败，不借用工具时间

IXsys 4.4.0，`--cuda-trace-scope process-tree --profile-from-start off`。无 GPU 上下文的父进程启动子进程，在限定采集窗口中执行 Graph 和图外 kernel。SQLite `cuda_graph_data`、`cuda_kernel_data` 与 `function_name_map` 用只读连接重新统计，源码 SHA 和备份 SHA 已核验。

V1 记录 20 个 Graph、20 个 `tianshu_stage_outside`，未记录图内 kernel。其断言把采集状态的单次 event 时间与停止采集后的 10 次时间比较，失败；首个采集 event 约 91 ms，不能解释为正常图耗时。原始失败和日志保留，未强制生成成功标志。

V2 修正测量设计，仍保留原定 5..20 缩放门槛和 .5..1.5 校准门槛。子进程输出正确，25 Graph（5 预热+20 测量）/20 图外 kernel 计数及进程归属通过，无采集状态 10/1 回放比值前后为 9.844162/9.845413。无采集单 Graph event 中位 0.119873077 ms；采集 trace 后 20 Graph 中位 0.426339500 ms，约为前者的 3.55659 倍，时长校准失败。图内 add/mul kernel 也未覆盖。因此完整能力检查仍为 failed；真实模型只采用独立通过的无采集 event 缩放证据，不启动 IXsys，不以其 trace 百分比判断当前瓶颈。

## 备份与离线复核

本机原始数据位于 `/home/czw_ubuntu_wsl/flagOS/service-results/20261016_tianshu/`：

| 目录 | 字节/分片 | 整包 SHA256 |
|---|---:|---|
| graphstage_v1/20261016_graph_stage_process_tree_tianshu | 47572 / 1 | `ad747d9251f7e43997245db573db58edc719279646c5ee7cddd16e7ce40f47a3` |
| graphstage_v2/20261016_graph_stage_process_tree_v2_tianshu | 49538 / 1 | `4630f6fa2fec933007d1858fe86ebfac67cbf60e4d25189c23543b7d0381eb16` |
| model_stage_events/20261016_model_stage_events_tianshu | 4282130 / 17 | `968f3168723d4151b186ae2400e11fe2cbb70ca636be36e7e6d8bff53c276b04` |

本机实际执行以下复核成功；检查 50 源码快照及当前仓库对应文件、39 实际环境项、75 预热点、所有 128 条负载响应和两个 canary、40 帧上下文与 Graph64、GPU 累计/中位/占比、CPU 中位、零 preemption 及硬件清理。V1/V2 原始失败状态保留，另存本地审计，不覆盖原结果。

```bash
python3 experiments/tianshu_graph_stage_profile_20261016/verify_downloaded.py \
  --cap-v1 ../service-results/20261016_tianshu/graphstage_v1/20261016_graph_stage_process_tree_tianshu \
  --cap-v2 ../service-results/20261016_tianshu/graphstage_v2/20261016_graph_stage_process_tree_v2_tianshu \
  --model-root ../service-results/20261016_tianshu/model_stage_events/20261016_model_stage_events_tianshu
```

本轮 GPU controller33216、API33218、Engine33373 均终态，未排队下一项 GPU 实验。报告与代码保存 Git；未改沐曦仓库，未推送远端。
