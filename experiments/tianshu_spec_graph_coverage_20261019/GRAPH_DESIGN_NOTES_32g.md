# 投机图扩展的源码约束（尚未实现）

以下仅来自当前安装源码，不是实际请求覆盖率结果。ModelRunnerFL SHA b0144bcff96bef15b04eb1ad45385b01c47faea056bc56352a6486b7c497fa2e；CudagraphDispatcher SHA 51f632753af729aeee6ce4a802d9d74b64f22eef02a524f36e2a24b07ef0cf63；实际 FL GraphWrapper SHA ba78ed0a736bb21a4f2c307fc9775f15e3742b94533eff333b0c364eb8213fc2（仓库与安装副本均核对）。

1. _is_uniform_decode 检查 maxQ==uniform_decode_query_len 且 tokens==maxQ*num_reqs。投机 1 token 时长度为 2；仅全 Q2 满足，不能据此宣称全 Q1 或混合批次的实际频率。
2. Dispatcher._create_padded_batch_descriptor 对 FULL uniform 以 padded_tokens//uniform_decode_query_len 计算 num_reqs。只 force_uniform=True 会给全 Q1 错误的请求数，不能复用 Q2 描述符。
3. _warmup_and_capture 并不直接把 desc.num_reqs 传给 _dummy_run；dummy 由 self.uniform_decode_query_len 和 token 数重新生成 query 列表。所以只往 key 集合塞额外 Q1 描述符还不够。若做 Q1 独立图，捕获、memory profile、运行调度三处都要按同一 query 长度生成正确元数据，并在 finally 恢复任何临时状态。
4. 混合 Q1/Q2 的完整图需要固定容量下可变 GPU cu_seqlens，同时保持 max_query_len=2、正确的请求数和物理 KV slot=-1 padding。不能把 mixed 标签改 uniform 掩盖它。仅用 token=128/request=64 大图可能给低提议率的批次带来过量 GEMM，需据实测 tokens/requests 分布选桶，计数器保留这两个维度。
5. 当前 native_verify 的 Q2 路由允许 n<=qrows<=2n、scratch 容量128，已有混合 attention 组件通过不等于模型图正确。需要新的多批次变更 input_ids/positions/cu_seqlens/block_table/seq_lens 的模型级数值与 canary、采样/接受数、logits 索引检查，再做服务与新 Math。
6. _dummy_run 默认 KV slot 全部 -1；真实 execute 在 FULL padding 时通过 _get_slot_mappings 显式填充无效 slot。dummy 的 no-write 不能代替真实 KV 生命周期正确性。

先按两个 4096/512 C64N64 诊断阶段统计真实 execute（含 prefill/纯 decode/verify）、实际 dispatch 与成功的 FL graph replay API 分支；profile/capture 单列，不把 Python kernel 编译调用次数当作图重放覆盖。观察器有 marker I/O/定期 JSON 成本，未测开销，不把诊断吞吐作为优化成绩。
