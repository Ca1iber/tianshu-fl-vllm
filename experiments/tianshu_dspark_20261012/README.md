# 天数 DSpark 兼容实验

目标是在天数 BI-V150 的 vLLM 0.24.0 环境使用用户已下载的 `/workspace/MiniCPM5-2B-DSpark`。当前属于开发诊断，未采用，尚无独立 Math-500 或正式吞吐成绩。

该 draft 是 5 层 Qwen3DSparkModel，hidden size 2048、全词表 130560、Markov rank 256、block size 7，读取 target 层 `[1,10,20,30,39]`。embedding 和输出 LM head 与 42 层 LlamaForCausalLM target 共享。原模型配置和权重保持只读。

安装的 vLLM 具有 DFlash 传输流程，没有 DSpark 模型注册。实验通过显式环境开关 `TIANSHU_DSPARK_COMPAT=1` 注册独立模型与 proposer，复用旧版 KV 管理及原始 rejection sampler。DSpark 的 7 个 query 包含 anchor，全部位置预测下一 token；每个 proposal 顺序加入前一个采样 token 的 Markov bias。固定深度为 7，加载 confidence 权重但尚未开启动态截断。仅支持此次 BF16、TP1/PP1、全词表模型。

诊断过程保留了模型注册、旧版 LM head 参数、缺失 CUDA RMSNorm、自定义 query-only RoPE、greedy SamplingMetadata 等失败日志。draft 专用 RMSNorm/RoPE 代理仅修改 DFlash 模块的引用，未覆盖全局 custom ops 或安装源码。query-only RoPE 的原地大 NeoX 样例未通过 CPU 检查，采用独立输出再复制后 12/12 RMSNorm/RoPE 及尾部检查通过；底层原因尚未证明。

`smoke_v6` 已成功完成 greedy 和概率采样各一项 96-token 原始补全文本请求。实际记录确认使用了 anchor-first 7-slot 布局和 Markov head；两项合计记录 36 draft blocks、252 draft tokens、162 accepted（64.29%），这些单路短样例不能作为吞吐提升证据。固定相同 eager target 参数的非投机对照中，greedy 第 3 个 token 出现分歧（` to` 与 ` not`）。因此暂停性能试验，先以固定前缀 logprobs 和原生多 token 注意力 CPU 检查定位差异。

质量采用比赛要求的独立 Math-500 Level3 严格大于 95%，不要求输出与旧版本逐字一致。但未知的适配错误必须先排查。当前正式最佳仍为 `tianshu-best-native-prefill-20261012`：2712.55 total tok/s、TTFT 4720.38 ms、102/105（97.14%）。

上游实现参考：[vLLM DSpark 模型](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_dspark.py)、[DSpark speculator](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py)、[MiniCPM5 DSpark 兼容讨论](https://github.com/vllm-project/vllm/issues/57013)。上游不同环境的接受率或性能不作为天数测量结果。
