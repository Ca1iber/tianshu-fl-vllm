# 天数 BI-V150 32g：BF16 descale 参数与 int32 地址正确性验证

真实服务的 `TritonAttentionImpl.forward` 在 BF16 / KVQuantMode.NONE 下也传入 expanded K/V descale 张量。原实验守卫要求两者为空，导致首次服务候选全部回退到 int64；该候选无有效路由，已停止，不用于性能结论。完整控制组三轮保留，后两轮均值为 2393.19494705931 total tok/s（16K/1K、C64、N64，本地实验）。

核对 native `_cast_kv_tile` 和 `USE_FP8_Q_DESCALE` 分支：KVQuantMode.NONE 时 K/V descale 的加载不参与浮点计算。守卫现在要求 BF16、量化模式 NONE、无输出 scale、偏移上界 < 2^31，允许未使用的 K/V descale；其它限制保留。物理指针和浮点计算未修改。

新增测试模拟实际交错 KV、Q stride 和 expanded K/V descales，10 场景全部通过 FP32 参考检查，且输出与原 int64 实现逐位一致。CUDA Graph 捕获/重放通过。以下仅为算子微测，不是服务收益。

| Batch | 最大上下文 | int64 ms | int32 ms | 速度提升 |
|---:|---:|---:|---:|---:|
| 1 | 5120 | 0.038763 | 0.038743 | 0.053% |
| 4 | 5120 | 0.130865 | 0.107487 | 21.749% |
| 8 | 5120 | 0.227616 | 0.203682 | 11.751% |
| 16 | 5120 | 0.466591 | 0.378331 | 23.329% |
| 24 | 5120 | 0.745587 | 0.712909 | 4.584% |
| 30 | 17408 | 2.822361 | 2.475233 | 14.024% |
| 32 | 17408 | 2.854297 | 2.531163 | 12.766% |
| 48 | 5120 | 1.173971 | 1.121738 | 4.656% |
| 64 | 5120 | 1.493796 | 1.486459 | 0.494% |
| 8 | 32768 | 1.351059 | 1.213150 | 11.368% |

验证原始归档：2054649 bytes；SHA256 `e9d41f7ee62331ff3303c847aec00d0cc93e185022bc02e7fde99c486c391d06`，所有分片与整包已在本地验证。
本地原始目录：`service-results/20261013_tianshu/attention_address32_bf16_descales/20261013_attention_address32_bf16_descales_tianshu`。

首次无效路由尝试完整保留：1705885 bytes，SHA256 `ed1090e2c9e1089ef8138d0f8cdabda33d96503dbf0b1144c7a427aef9ba1282`；本地 `service-results/20261013_tianshu/attention_address32_service_stopped_v1`。其停止审计与不完整候选数据保留，不补造完成结果。

修正后的服务测试已实际捕获 B4/B8/B16/B24/B32/B40/B48 的 int32 路由，模式 NONE，K/V descales 非空，偏移上界通过。服务对照还在进行，未采纳此配置，尚未进行匹配的新 Math-500 或正式 benchmark。正式最好成绩仍为 4K 3222.52、16K 2440.62 total tok/s。
