# 原生 INT8 GateUp 服务短测

接入版本固定为 tianshu_native_int8_prefill_service_v2_20261022。先要求独立 smoke 完成，再串行 control/candidate：4K 输入/1K 输出，C64/N64，各三轮排首轮，原模型只读，原 serve 13 项参数不变，两臂 877 份冻结源相同，原 471 份精确子集；新增缓存环境明示，总共观测 63 项环境。

两臂都预热原生库且由模型 DeviceMemoryProfiler 统计。candidate 仅 M512/4096 原生 _int_mm；两臂 M8/M64、LM 和 KV 等保持相同。profile 必须证明 candidate 全42层 M4096 在原生 memory_profiling_fl 内实际运行，临时工作区计入预算；不得手调 KV 大小。原 attention 与新 INT8 记录分开。

独立 CPU 重读全响应和日志，重算请求数、tokens、TTFT、吞吐、后两轮差幅，并重跑完整运行时审核；性能门槛 >0.5% 且每臂差幅 <1%，无就绪后 autotune，无抢占/错误。通过后才做全新 Math105 C8 与真实 C64；C8 未命中新尺寸必须明确记录，不将其冒充新路径的质量证据。正式 N256 也必须新跑，不借用组件或历史成绩。

每天北京时间 06:00–24:00，当前硬停止 2026-10-22 23:58；GPU 阶段失败停止流水线，保留原结果和冻结源码，修复另建版本。

## 编排 V2

原 smoke 的 GPU 请求正常，但通用三轮审计要求 >=194 次请求，实际 smoke74次触发失败。原结果保持 audit_failed；独立 CPU_RECOVERY_GATE 只验证74个响应和真实新路径、显存预算、binary与原M64 capture，不将 smoke 当成性能/质量通过。编排V1仅等候时失败，两臂GPU未启动。本V2以该明确范围的独立接入核验为前提，三轮A/B仍保留完整原通用审计及其194请求门槛。
