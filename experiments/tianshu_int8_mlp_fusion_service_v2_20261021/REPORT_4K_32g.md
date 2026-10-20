# 整数 GateUp + SiLU 融合 V2：小 4K 服务没有可采用收益

本次不采用 V2。融合候选有效均值 **5247.420604 total tok/s**，比同轮新边界 control 高 **0.295527%**，但比已合格版本的同客户端小 4K **5410.385242** 低 **3.012071%**。两个有效候选轮次的极差/均值 **1.158403%**，control 为 **1.324690%**。没有安排新的 Math 或正式 N256；不能将组件快 13.67% 写作模型吞吐收益。

## 工作负载与逐轮成绩

原 13 项 serve argv，C64，4096 输入 / 1024 输出，每轮 64 请求。每臂 3 轮，剔除第 1 轮，后两轮平均。所有 384 个计时请求成功，逐请求 usage 精确等于 4096/1024/5120。没有完整 16K。

| 模式 | 轮次 | Total tok/s | Output tok/s | TTFT ms | autotune 数 | 计入 |
|---|---:|---:|---:|---:|---:|---|
| control | 1 | 4950.325079 | 990.065016 | 13899.320019 | 0 | 否 |
| control | 2 | 5197.305148 | 1039.461030 | 11628.803568 | 0 | 是 |
| control | 3 | 5266.612392 | 1053.322478 | 11641.640616 | 0 | 是 |
| candidate | 1 | 4935.709319 | 987.141864 | 13928.050428 | 0 | 否 |
| candidate | 2 | 5217.027478 | 1043.405496 | 11637.024115 | 0 | 是 |
| candidate | 3 | 5277.813730 | 1055.562746 | 11623.586144 | 0 | 是 |

control 均值 5231.958770，TTFT 11635.222092ms；candidate 均值 5247.420604，TTFT 11630.305130ms，TTFT 差 -0.042259%。六轮均无运行时 autotune，因此本次波动不能归因于已观测到的稀有 shape 临时调优。

## 真实执行与比较范围

- 两臂相同 568 项冻结运行时源码、63 项实际环境、原 serve argv。43 份 INT8 权重和 scale 与合格 Math 版本的签名逐一相同，原 BF16 权重保留并计入真实加载内存分析。
- 实际模型为 LlamaMLP，42 层均绑定；原 Down projection 保留。M8/M64 的融合 capture 与模型 replay 都逐层验证，C64/4K 至少 100 次 replay。两种 M 的实际融合二进制 SHA 都与组件测试匹配，寄存器溢出为 0。
- control GateUp M64 配置为 BM64/BN256/BK128/4 warps/1 stage，与组件原生基线一致。候选融合配置为 BM64/BN64/BK128/2 warps/1 stage。不能将模型差异解释为组件拿错了 GateUp 基线配置。
- 模块实际选择的激活是 `SiluAndMul.forward_native`。两臂都增加了新的 MLP custom-op 边界，control 和候选非融合 M 在内部执行该 native 激活；融合 M8/M64 使用已逐位验证的 FlagGems 标量计算。组件逐位一致的证明范围是组件原生 FlagGems 基线，不是先前整个编译模型的所有激活边界。V2 未跑新 Math，也未继承旧正确率。
- 两臂原始实际 KV blocks 分别 **40820 / 40690**，没有 preemption。不能把它们写成相同缓存容量；内存与 capture 实测字节已保存在 SCREEN_GATE。

## 结论与后续

实际融合执行成立，但端到端没有稳定收益，且新边界版本整体低于原合格同客户端参考。下一实验 V3 将两臂未融合形状的激活显式改为单趟 FlagGems，以检验能否减少新边界的开销；这仍是待验证假设，涉及激活计算边界变化，需要新的 Math 和正式成绩才能采用。

V1 在首轮前主动中止的原因记录在 V1/STOP_REPORT_32g.md。V2 是全新两臂测试，不复用 V1 数据。已采用的正式最好版本仍为 **5482.79 total tok/s，Math 101/105**，tag `tianshu-original-command-4k-lm-bf16-prefill-20261021`。

原始容器目录：
- `/workspace/benchmark-results/20261021_int8_mlp_fusion_v2_service_control_tianshu`
- `/workspace/benchmark-results/20261021_int8_mlp_fusion_v2_service_candidate_tianshu`
- `/workspace/benchmark-results/20261021_int8_mlp_fusion_v2_pair_runner_tianshu`

独立复核：`verify_pair.py` 验证所有请求、全部冻结源码及快照、实际进程环境、权重签名、运行路径、重新读取的 runtime audit 与已保存 audit 完全一致。SCREEN_GATE/SCREEN_ROUNDS 仅保存关键结果；完整响应与日志保留在容器。
