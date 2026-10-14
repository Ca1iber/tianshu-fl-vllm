# 天数 BI-V150 32g：KV16 单 warp 独立五轮复测

基于当前正式最好配置的独立对照：Decode Attention 对 batch 8/16/32/40/48/56/64 使用 KV tile16、warp1、stage1、L0；batch24 保留 KV32/warp2。生产者与 reducer 使用相同 tile，地址宽度不变。原生 Prefill M256/KV128/warp16/L0、75 点预热、logits gather、budget2048、TopP16 保持一致。

## 方法与结果

4K 输入、1K 输出、并发64、每轮64请求，双方各五轮。请求 temperature0/top_p1/repetition_penalty1/ignore_eos，匹配正式脚本的贪心负载；仍属于小规模服务实验。预先规定排除第1、2轮，统计第3～5轮。原 foundation 保留的仅排除首轮统计与本次 confirm_scoring 分开。

| 组别 | 轮次 | Total tok/s | Output tok/s | TTFT ms | 秒 |
|---|---:|---:|---:|---:|---:|
| control | 1 | 1377.4004 | 275.4801 | 13403.671 | 237.8974 |
| control | 2 | 3846.5035 | 769.3007 | 13061.140 | 85.1891 |
| control | 3 | 3843.1125 | 768.6225 | 13130.380 | 85.2642 |
| control | 4 | 3846.7068 | 769.3414 | 13034.629 | 85.1846 |
| control | 5 | 3844.6640 | 768.9328 | 13075.844 | 85.2298 |
| candidate | 1 | 1390.8806 | 278.1761 | 13335.682 | 235.5918 |
| candidate | 2 | 3929.4539 | 785.8908 | 12990.394 | 83.3907 |
| candidate | 3 | 3926.6407 | 785.3281 | 13049.803 | 83.4505 |
| candidate | 4 | 3926.8560 | 785.3712 | 13037.859 | 83.4459 |
| candidate | 5 | 3927.5666 | 785.5133 | 13008.912 | 83.4308 |

计分均值：对照 3844.827775 → 候选 3927.021103 total tok/s，提升 **2.137764%**。
control 计分轮极差/均值 0.093485%；候选三轮全部快于对照三轮。
candidate 计分轮极差/均值 0.023579%；候选三轮全部快于对照三轮。

## 核验范围

本地下载核验了全部640份请求的 usage、seed、文本、TTFT 和吞吐重算；双方320对响应中286对文本相同，7个稀有shape canary相同。输出并非全部相同，必须独立验证新配置的数学正确率。核验7份冻结源码与当前仓库、双方12份安装文件快照、真实启动参数、双方75点完整预热，以及实际模型 B64 KV16/warp1、B24 KV32/warp2 回退及 producer/reducer tile 一致。

原始备份 3,409,429 bytes，14份分片逐份及全包 SHA256 验证。全包：`46134c3b60f5c2143fd8191665b57f05617b1c5192c4e9a4eefc4627ef13fcd4`。完整原始记录位于工作区 `service-results/20261014_tianshu/decode_kv16_w1_confirm5/20261014_decode_kv16_w1_confirm5_tianshu`。

核验命令：

```bash
python3 experiments/tianshu_decode_kv16_w2_20261014/verify_downloaded_confirm.py --root /path/to/20261014_decode_kv16_w1_confirm5_tianshu
```

## 决策

通过进入独立 Math-500 Level3 的门槛，尚未采纳为正式最好配置。新105题评测已单独启动，等待真实结果；通过 >95% 后再运行原始4K N256四轮 benchmark。当前正式最好仍是 KV32/warp2 配置的3640.16 total tok/s，旧103/105正确率不用于证明新配置。未进行16K服务评测。
