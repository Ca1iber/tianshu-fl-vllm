# 天数32g：关闭前缀缓存没有吞吐收益

同期新控制与候选各3轮4K/1K、C64/N64，预先排除首轮。计分均值 **4067.843005 → 4067.641697 total tok/s，−0.004949%**。不进入五轮、新Math或正式benchmark，保留前缀缓存开启与现有最好3826.24。

| prefix cache | 轮次 | Total tok/s | TTFT ms | 成功 | 计分 |
|---|---:|---:|---:|---:|---|
| 开 | 1 | 1408.689085 | 13374.284793 | 64 | 否 |
| 开 | 2 | 4068.558135 | 13014.313166 | 64 | 是 |
| 开 | 3 | 4067.127876 | 12998.914107 | 64 | 是 |
| 关 | 1 | 1395.108281 | 13265.529760 | 64 | 否 |
| 关 | 2 | 4067.523647 | 13003.884563 | 64 | 是 |
| 关 | 3 | 4067.759747 | 12994.693594 | 64 | 是 |

双方同为B64S8、页长16、guardedKV16/warp1、原生Prefill L0、budget2048/threshold0、75点预热与gather。server CLI只差显式enable-prefix-caching与no-enable-prefix-caching。实际Engine日志的True/False逐组核验，不能只以命令行当作实际配置证据。producer/reducer、B64S8与B24回退、完整75次预热保持一致。

384份响应逐一核对usage4096/1024/5120、种子、文本、TTFT、耗时与吞吐重算，7份冻结源码及双方12份安装快照均通过独立本地复核。169/192对文本相同，7个greedy canary相同；不声称所有输出一致，不替代随机采样Math。

SSH观察连接在12点附近被远端关闭，重新连接后原控制器正常complete，全部GPU子进程已结束，候选第3轮结果完整。没有重启、补跑或替换该轮，随后串行启动独立Decode padding对照。不能由观察连接中断推断容器/测试重启。

原包3,446,970bytes/14片，SHA `ed369845a1629687a4c0d3f0af16a1c203baa808e8ff43cc1db14fa4c1cc7f51`，原包/分片SHA完整核验。原件service-results/20261014_tianshu/prefix_cache_S8_service/prefix_cache_service。

```bash
python3 experiments/tianshu_prefix_cache_20261014/verify_downloaded_service.py --root /path/to/prefix_cache_service
```

没有新正确率或16K结论，安装源/官方脚本/沐曦内容未修改。
