# 天数 BI-V150 32g：原始官方命令独立 Math105

本次新启动服务，逐项使用用户提供的原始官方 serve 命令；不沿用 N64 小测的服务或 KV 缓存，不借用之前自选调度参数的准确率。真实 Worker 为默认每步 2048 tokens、256 seq、async=True、模型最大长度 131072，实际预热 75 点。

保持已验证的 CV Decode 缓存策略、KV16/warp1、B64 S8、native Prefill L0、预算内 padding/int32 gather、原生 TopP 和原采样器。未添加量化或投机。Math 的实际请求并发为 8；B64 路由另由服务吞吐实验验证。

使用用户给出的 EvalScope Math500 本地 Level 3 子集，共 105 题、并发 8、temperature=1.0、top_p=0.95、max_tokens=32768、timeout=3600。工作目录独立，以保存完整原始产物。

结果为 **103/105，98.095238%**；105 请求全部成功、0 错误，超过 95% 要求。错误题为零基索引 18、87。104 请求自然 stop，索引 83 达到原定 32768 token 上限且评分正确；没有人为截断后续请求。服务 metrics 无 KV preemption。

105 条原始 predictions、105 条 reviews、EvalScope report 与退出状态均在本机重新解析核对；44 份运行源及源码快照、真实进程的 35 项算子配置、SDK 默认值、75 点预热和实际算子路径一并验证。FlagGems 提交为 4cdc3b760205bb4ce7edd9099fa19fa4986adc24，观测时干净。模型 config 与 generation_config 元数据 SHA 已记录；本次未对全量模型权重重新做哈希。

原始归档 4,630,101 bytes / 18 parts，SHA256 `8557d8a46a3b269926bc91c430bb1c9768ed0e882e57bd65601d4737ec0d5638`。归档中的仓库 HEAD 为 `b872152d8e4d9fff9cb7744f399431118975c385`；用户原有 GateUp 两文件工作区改动保留，未参与本 profile。

本机 raw：`service-results/20261016_tianshu/original_command_math/20261016_original_command_math_tianshu`；远端：`/workspace/benchmark-results/20261016_original_command_math_tianshu`。复核命令：

```bash
python3 experiments/tianshu_original_command_20261016/verify_downloaded.py --root <MathRaw>
```

此质量结果只绑定本报告的原始命令、35 项算子配置和 44 份冻结源码。随后已在新服务中启动相同配置的正式 N256 4K 四轮验证；当前尚不声明新正式吞吐成绩。N64 重复输入的缓存成绩不能代替该验证。本轮未运行 16K，也不凭单一 4K 承诺整套比赛资格。
