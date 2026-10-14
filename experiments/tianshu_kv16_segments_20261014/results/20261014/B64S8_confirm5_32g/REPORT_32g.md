# 天数 BI-V150 32g：B64 八分段的独立五轮确认

同期新控制与候选各5轮4K输入/1K输出、并发64、每轮64请求。预先排除第1/2轮，后3轮均值 **3928.414099 → 4065.538627 total tok/s，+3.490582%**。这是N64小规模服务测试，不能作为原版N256正式成绩。

| 组别 | 轮次 | Total tok/s | Output tok/s | TTFT ms | 秒 | 计分 |
|---|---:|---:|---:|---:|---:|---|
| control | 1 | 1383.2592 | 276.6518 | 13346.857 | 236.8898 | 否 |
| control | 2 | 3931.0239 | 786.2048 | 12990.250 | 83.3574 | 否 |
| control | 3 | 3928.5415 | 785.7083 | 13026.566 | 83.4101 | 是 |
| control | 4 | 3927.7721 | 785.5544 | 13049.612 | 83.4264 | 是 |
| control | 5 | 3928.9287 | 785.7857 | 13015.805 | 83.4019 | 是 |
| candidate | 1 | 1409.8045 | 281.9609 | 13275.141 | 232.4294 | 否 |
| candidate | 2 | 4065.5859 | 813.1172 | 13082.357 | 80.5985 | 否 |
| candidate | 3 | 4064.6248 | 812.9250 | 13062.363 | 80.6175 | 是 |
| candidate | 4 | 4064.9735 | 812.9947 | 13055.652 | 80.6106 | 是 |
| candidate | 5 | 4067.0176 | 813.4035 | 13002.445 | 80.5701 | 是 |

control计分极差/均值：0.029441%。

candidate计分极差/均值：0.058855%。

候选所有计分轮均高于对照所有计分轮：True。双方不复用旧结果。控制与候选都启用guarded KV16/w1；唯一更改是B64 adaptive segments4→8，其他批次保持原策略。B24保留KV32/w2、4分段；小批次保留原w4。地址宽度、原生Prefill M256/KV128/w16/s1/L0、预算2048/threshold0、TopP16、75点预热与int32 gather、BF16不变。实际模型producer及reducer配置成对核验，不以仅导入hook作为启用证据。

640/640请求实际usage4096/1024/5120、种子、文本、TTFT与吞吐重算通过。320对响应中278对文本相同，7个greedy canary相同；不声称全部输出一致，不替代新官方随机采样Math。7份冻结源码及双方12份安装源码快照、相同server argv、75点完整预热、B64S4/S8与B24回退核验通过。

原始包3,512,710bytes/14分片，全包SHA256：`45bce59fccf40081730177a701094423ba0bdaab1061aa4c7dccecbd2e9bc926`。原件在工作区`service-results/20261014_tianshu/kv16_segments_B64S8_confirm5/20261014_kv16_segments_B64S8_confirm5_tianshu`。分片、全包SHA及本地完整原始响应均已核验。

复核：

```bash
python3 experiments/tianshu_kv16_segments_20261014/verify_downloaded_confirm.py --root /path/to/20261014_kv16_segments_B64S8_confirm5_tianshu
```

新Math105 runner/auditor及正式4K runner/auditor/独立离线核验器已准备。本检查点已启动新的Math105（runner155077），结果待定；正式4K尚未运行。不可沿用旧KV16/S4的102/105。新Math使用独立fresh_math_for_B64S8_segments标记、5份命名空间源码快照、官方batch8/temperature1/top_p0.95/max_tokens32768参数。官方Math C8不代表B64问答覆盖；B64数值微测与真实4K服务绑定证据独立保存。

正式最好保持3701.31 total tok/s、TTFT4464.89ms、旧配置Math102/105、tag tianshu-best-decode-kv16-w1-20261014。只有新Math严格>95%后才跑原版4K N256四轮预排首轮，且新正式成绩超过3701.31才更新最好版本。无16K、无沐曦源码修改、无推送。
