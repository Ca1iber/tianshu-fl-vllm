# 天数 BI-V150：Down prefill INT8，r2

保持原始 MiniCPM5-2B 文件和结构，只在 M128/512/4096 为 Down 使用 native A8W8。M64 和其他尺寸保留原 BF16 路径；已发布 r1 的全部 877 个文件保持原字节。新版本独立保存，没有覆盖旧 r1。

正式 4K（4096/1024/C64/N256，4 轮剔除第 1 轮）：**5911.45 total tok/s**、**1182.29 output tok/s**，平均 TTFT **3305.80 ms**，后三轮最大差距/均值 **0.2008%**。相对上一版正式 5777.22 total tok/s 提高 2.323%，TTFT 相对 3459.44 ms 变化 -4.441%。

历史正式比较有时段差异；本次另做了相同输入、精确 r1 基准的 N64 短对照，提升 **2.0897%**，候选热轮波动 0.0724%。

新鲜完整 Math-500 Level 3：C8 **102/105（97.14%）**，C64 **103/105（98.10%）**，原始逐条评审已独立重算。C8 本轮未命中 INT8 Down；C64 已验证 M128/4096 全 42 层真实执行，M64 图仍 BF16。达到输出上限的请求也全部计分。零抢占、无服务就绪后的 Triton autotune、所有源文件/控制器快照和继承的运行检查通过独立核验。

额外 INT8 Down 权重和缩放约 504.33 MiB，原 BF16 权重保留。对照短测的实际 KV 容量从 642784 降至 621440 tokens，分配与预热纳入模型内存评估。没有重跑 16K，不能把此结果当作完整双项比赛成绩。

在天数容器中启动：

```bash
cd /workspace/vllm-plugin-FL
bash experiments/tianshu_native_int8_down_prefill_release_v2_20261022/restore_generated_sources.sh
bash experiments/tianshu_native_int8_down_prefill_release_v2_20261022/restore_frozen_cache.sh --check
bash experiments/tianshu_native_int8_down_prefill_release_v2_20261022/restore_controller_evidence.sh --check
TIANSHU_NATIVE_INT8_PROFILE_DRY_RUN=1 bash experiments/tianshu_native_int8_down_prefill_release_v2_20261022/serve_tianshu_minicpm.sh
bash experiments/tianshu_native_int8_down_prefill_release_v2_20261022/serve_tianshu_minicpm.sh
```

若六份控制证据 JSON 缺失，可运行 `restore_controller_evidence.sh` 无参数，仅从随版 gzip 恢复缺失的精确字节，已存在但哈希不符时拒绝覆盖。这些是冻结配置和路由审计，没有上传题目、预测全文、数组或日志。实际模型文件、依赖环境及编译缓存留在容器；运行前逐项验证。所有正式 benchmark 参数保持原始 13 参数，恢复 63 项环境，仅重定向记录输出路径并清理继承的实验变量，不启动额外参数配置。

发布配置核验 890 个模型输入和 44 个控制器输入，启动前必须全部哈希一致。完整关键成绩见相邻 `tianshu_native_int8_down_prefill_official_results_v2_20261022`，质量见 `tianshu_native_int8_down_prefill_quality_results_v2_20261022`，短测见 `tianshu_native_int8_down_short4k_v2_20261022`。

完整 CPU comparison 以相邻 results 目录中的 `CPU_COMPARISON_32g.json.gz` 保存，解压字节哈希与正式 gate 一致。未压缩原件只留容器，不是服务启动输入。
