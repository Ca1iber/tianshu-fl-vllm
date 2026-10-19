# 混合投机批次补齐：原模型校验失败记录

2026-10-19，BI-V150 32 GiB。原实验提交 `443c7fc`，结果目录
`/workspace/benchmark-results/20261019_ragged_graph_model_probe_tianshu`。

这是未采用的失败实验，没有完成吞吐轮次，也没有新的 Math-500 结果。

三个请求数保持 64、仅补齐 token 的样本通过原数值门槛；第一个请求数
63→64、token 数 78→80 的样本失败。候选图与同样补齐的 eager 输出逐位一致。
与完全未补齐的 eager 对照相比，hidden 的最大绝对差为 2.0，43,181 个元素
超出 `atol=.02, rtol=.01`；logits 最大绝对差为 .4140625，2,574,902 个元素
超出同一门槛。仅据此尚不能判定具体算子错误或最终准确率。

三个通过样本及失败样本均核验了真实 token 的位置、query 边界、block table
和 slot 映射；42 层候选 KV 页已恢复，抽查的非目标页未改动，CUDA RNG 状态
保持一致。未补齐对照的 attention 请求数与 GEMM token 尺寸均可能改变，
后续使用独立提交的逐层观察代码定位首次差异，不修改原失败记录或放宽门槛。

原始归档 4,834,208 字节，SHA-256：
`f8c2f3a6e86d5e685ae03dd2be0351979357bf6faa06b72d5f62bcc0aa0f0a85`。
本地已校验 95 份源码快照及原失败状态，保存于
`service-results/20261019_tianshu/ragged_model_failed/`；离线核验结果为
`service-results/20261019_tianshu/failed_model_download_audit.json`。
归档包含全部现有原始文件，不重建失败前未被客户端保存的响应，不将未完成
的 128 请求记作成功。归档时 GPU 已空闲。
