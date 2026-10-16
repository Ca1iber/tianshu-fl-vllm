# 天数 BI-V150 32g：原始官方命令正式 4K 与独立质量验证

原始官方启动命令下，正式 4K 后三轮平均 **3992.97 total tok/s、798.60 output tok/s、TTFT 4443.93ms**。对应新独立 Math105 为 **103/105（98.095238%）**。本版作为“原始命令已验证 4K 配置”保存。

服务命令逐项等于用户提供的官方命令，实际 Worker 保持默认预算 2048、256 seq、async=True、stream interval 1、最大模型长度 131072。未增加调度预算/maxseq/stream/threshold 参数，未在源码里改写这些默认值。保留 CV Decode、KV16/warp1、B64 S8、native Prefill L0、75 个实际 Prefill 预热点与预算内 padding/int32 gather、原生 TopP 和原采样器；BF16、page16、prefix cache 开启。

使用未修改的官方 benchmark 脚本，4K 输入/1K 输出/C64/N256，共四轮，按原脚本预先排除首轮。模型 generation_config 中 temperature=1.0、top_p=0.95；原 benchmark CLI 使用模型默认采样配置。

|轮次|total tok/s|output tok/s|TTFT ms|耗时 s|成功请求|
|---|---:|---:|---:|---:|---:|
|1（排除）|3909.88|781.98|5931.27|335.23|256.0|
|2|3996.8|799.36|4438.77|327.94|256.0|
|3|3998.48|799.7|4444.43|327.8|256.0|
|4|3983.63|796.73|4448.58|329.03|256.0|

后三轮总吞吐极差/均值 **0.3719%**。服务启动完成后打印的 tuning/autotune 记录为 0 条；此统计不等同于测量了所有潜在编译成本。最终 prefix cache 查询 4194304 tokens、命中 12240 tokens，累计命中率 0.291824%。所有 metrics 的 KV preemption 为 0。N64 重复输入小测的 5510.08 不纳入正式成绩。

相对公开 4K 基线 2028.01 total tok/s，本版提升 **96.891041%**。TTFT 4443.93ms 低于公开基线 11573.53ms 的 +1% 上限 11689.2653ms。该条件及性能指标来自公开赛题 https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn ，本日保存的公开 API 内容 SHA 为 b711f5eb220c0412664647b34467a46d92a5912c419024699125f50116b81ba6。

质量服务与吞吐服务独立启动，44 份冻结源码和 35 项实际进程算子配置逐项一致（记录目录重映射）。Math Level3/C8/105题、temperature1/top_p.95/max32768/timeout3600，105成功0错误；104自然stop，索引83到上限且评分正确，错题18、87。所有105 predictions/reviews/report 已本机重算，详见相邻 math_32g 报告。

本机复核了四轮原始 stdout/CSV 的1024成功请求、各轮 token 总量、原 benchmark SHA、44份源快照、实际2048/256/75预热、CV/KV16/S8/原生TopP/原采样及零preemption；脚本未保存逐请求正文，不声称验证了不存在的响应正文。实际 shell dry-run 已逐项对齐完成 Math 和正式服务的 argv/35环境，证明文件 launcher_actual_binding.json；对应可用启动脚本是 experiments/tianshu_original_command_20261016/serve_tianshu_minicpm.sh。

正式原始归档 3887184 bytes / 15 parts，SHA `163c6c8e1c05868764928c963ca822a6f8f56e72a18827146fe16baa1a5e2149`；新 Math 归档 SHA `8557d8a46a3b269926bc91c430bb1c9768ed0e882e57bd65601d4737ec0d5638`。分片与整包 SHA 均通过。FlagGems 仓库观测为干净 4cdc3b760205bb4ce7edd9099fa19fa4986adc24；模型两份元数据 SHA 有记录，全量权重未重新哈希。

raw 位于 service-results/20261016_tianshu/original_command_official/20261016_original_command_official_tianshu，远端同名 benchmark-results 目录；复核：

```bash
python3 experiments/tianshu_original_command_20261016/verify_downloaded.py --root <OfficialRaw> --math-root <MathRaw>
```

历史4091.29使用额外调度参数，保持其原始成绩和标签；与本版启动条件不同，不据差额判断算子收益。今天的自选参数组合确认曾取得+2.06065%的小测收益，但未作为本版新质量/正式配置采用，相关报告在 tianshu_combined_20261016。本版未运行16K，完整比赛资格与总赛分仍需各场景及复现要求的后续验证。
