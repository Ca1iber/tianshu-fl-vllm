# 原生 INT8 服务独立质量与正式评测计划

只有同源 4K N64 三轮对照收益 >0.5% 且差幅 <1% 后才启动。使用 service_v2 原始 877 份冻结源码；所有旧 471 数值源码精确保留。服务环境 62 项和显式 TRITON_CACHE_DIR 总 63 项逐项通过 /proc 观察；原始官方13项启动 argv 不变。所有验证控制器及递归导入的实验模块单独快照和散列。

C8：官方 Math500 Level3 全105，eval batch8，温度1/top_p.95/max_tokens32768，>=100/105。保留原 C8 模型精度、attention、采样、KV、全部43权重bank与M8/M64图检查；新路径未命中时如实报告0，不声称验证了新路径。

C64：另跑全105，仅 TIANSHU_FUTURE_KV_MODE 从 candidate 改 control，恢复原生请求准入，native allocator 本身不改。继承实际C64图重放>=100次、42层M64融合、LM INT8和长上下文attention检查，并必须在真实 Math 请求下覆盖全部42层原生M4096预填充。所有输出实际 review 成功，无缺失/接口错误，>=100/105。

新服务结构检查保留原生 profile 中所有42层M4096命中与显存预算证据，全部原本23尺寸完整binary匹配、新原生binary匹配、临时workspace、无新增权重payload、全部FX改写、RNG和源码冻结。结构帮助程序自身不冒充完整审核；完整旧检查另由相应C8/C64质量审核保留。

两份新质量通过后，使用官方原始 benchmark 脚本做4K/1K C64 N256四轮排首轮。CPU重读原CSV、请求数、tokens、TTFT和吞吐，绑定两份质量来源与相同877源码/63环境（C64唯一准入差异明示）。新成绩不借用历史5695.95；此数只作历史比较，性能收益以同源A/B为准。失败均保留原实验，不改冻结源，修复新版本。

当前运行窗口至2026-10-22 23:58硬停止；每日06:00–24:00使用。

## V2：按真实 C8 路径复核，保留原失败

首次新C8 raw全105题101分、0接口错误，但原完整审计要求active INT8预填充42层。实际19种运行M为48/56/72/80/88/96/104/112/120/136/144/152/160/168/240/272/288/320/352，均不在预填充INT8白名单，无新M512/M4096、无旧M范围融合。V1原GPU状态audit_failed保持不动，控制器与源码也不改。

V2不改任何877份模型数值源码。C8质量检查保留旧attention/KV/scheduler/sampling/43bank/M8 NN/M64 capture/LM8 BF16等完整检查，另外要求真实BF16回退覆盖全42层、所有委托M不在任何INT8白名单、所有输入stride和调用有效。明确actual_new_prefill_path_all_42_GateUp_verified=false，不虚构INT8预填充命中。

单独CPU恢复目录核对当前新C8全部105份prediction/review与原始report，绑定原GPU audit_failed结果hash、冻结877源与控制器，并输出CPU_recovery_complete。它不是原GPU实验通过，也不是旧准确率背书。后续官方控制器用该实际raw_root做环境绑定。C64要求仍严格：同877源、全部105新请求>=100正确，真实M4096原生INT8全42层、真实M64图与LM64 INT8和全部原检查；只有C64通过才允许正式N256。
