# 天数32g：Decode 图尺寸合并没有稳态吞吐收益

当前 B64S8 配置下同期新控制与候选各3轮，4K输入/1K输出、C64/N64；预先排除首轮。计分均值 **4065.557857 → 4065.905761 total tok/s，+0.008557%**，双方计分轮次重叠。不进入五轮、新Math或正式benchmark，保留最好3826.24。

候选只移除24/40/48/56四个Decode图尺寸，配置75→71，实际捕获35→31；运行时向上补齐至保留的图。原有完整75点Prefill预热、最大2048及全部大于64的尺寸保持不变。两组同用记录hook，控制不移除尺寸。这个实验与先前51→75点预热不同；不否定75点预热避免临时调优的作用。

| 配置 | 轮次 | Total tok/s | TTFT ms | 计分 |
|---|---:|---:|---:|---|
| control | 1 | 1387.135533 | 13345.198918 | 否 |
| control | 2 | 4067.953747 | 13026.494199 | 是 |
| control | 3 | 4063.161967 | 13071.818193 | 是 |
| candidate | 1 | 1315.129388 | 21354.208961 | 否 |
| candidate | 2 | 4066.241398 | 13072.527300 | 是 |
| candidate | 3 | 4065.570123 | 13070.154377 | 是 |

384份响应逐一重算usage4096/1024/5120、种子、TTFT、耗时和吞吐；8份冻结源码与双方12份安装快照核验通过。实际FULL Decode分派在吞吐阶段marker后记录，逐条验证向上补齐、原dispatcher源码SHA、producer/reducer一致、B64S8和B24回退、75次Prefill预热。166/192对文本相同、7个greedy canary相同；没有新Math结果，不声称所有生成文本一致。

控制/候选首轮耗时236.227818/249.161796秒，均预先排除，不拿冷轮差异当吞吐收益。首轮TTFT候选更高也完整保留。

原包3,477,976bytes/14片，SHA `54ca594c8d329a7c4db2e6b63a9d266f55db5404587886e68330761b15dbf2af`，整包和分片均校验。原件 `service-results/20261014_tianshu/decode_padding_S8_service/decode_padding_service`。

复核：`python3 experiments/tianshu_decode_padding_20261014/verify_downloaded_service.py --root /path/to/decode_padding_service`。安装源、官方benchmark与沐曦仓库未修改，无新16K成绩。
