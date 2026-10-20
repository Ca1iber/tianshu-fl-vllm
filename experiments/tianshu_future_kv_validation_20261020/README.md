# 未来输出KV准入：独立质量验证与正式评测

最终候选仅纳入未来输出KV准入，在128MiB暂存区、max_num_seqs64的对照背景上，16K/1K C64 N64筛选3737.45→3887.60 total tok/s（+4.0174%），抢占6→0。整数元数据融合4K服务增益仅+0.00973%，排除该组件。

最终运行来源固定261份、配置环境52项，与完成的16K筛选逐项一致。运行来源与本目录控制器分别冻结；额外7份控制器文件另存controller_sources.json及来源快照，不将其冒充16K筛选的运行来源。原始serve命令、官方benchmark脚本及模型、采样配置保持评测要求。

CPU_GATE_32g.json只记录CPU准备和对已完成筛选记录的兼容性审计，不代表新Math或正式评测通过。2026-10-20 17:32北京时间实际状态为新Math运行中；通过严格>95%后才启动正式评测。不可复用旧版102/105的质量结果，也不可将此前260来源4K筛选描述为当前261来源的正式成绩。

执行入口run_math_and_official.sh：

1. 新Math-500 Level3全105题，原evalscope命令，C8、temperature1、top_p0.95、max_tokens32768。核对唯一题目、完整预测、无错误、准确率严格>95%，及真实运行绑定与最终KV计划全部释放。
2. 仅质量通过后，运行原官方脚本4轮4K/1K C64 N256、4轮16K/1K C64 N128，分别预先排除首轮。完整1536个请求必须成功；与本次Math匹配全部261来源、52项归一化环境和7份控制器。
3. 质量与正式结果分别写入新目录，保留原始输出。最终采用与否仍需独立复核正式4K无明显退化及16K增益；脚本完成本身不自动更新正式发布。

原始输出：

- /workspace/benchmark-results/20261020_future_kv_math_tianshu
- /workspace/benchmark-results/20261020_future_kv_official4k16k_tianshu
- /workspace/benchmark-results/20261020_future_kv_validation_pipeline_tianshu

GPU截止北京时间2026-10-20 23:58，控制器设置绝对时间闹钟，并在finally终止服务和评测进程组、确认GPU空闲。普通后续探索仍先用4K N64；完整双工作负载用于这次长上下文准入优化的最终定版。
