# 天数32g：页长32服务仅提升0.12%，不采用

B64S8最好配置下，同期新控制page16与候选page32，各3轮4K/1K、C64/N64，预排首轮。计分均值 **4067.454766 → 4072.274965 total tok/s，+0.118507%**。达不到2%筛选门槛，不做独立五轮/新Math或正式benchmark，保留page16正式最好3826.24。

| 组别 | 轮次 | Total tok/s | TTFT ms | 成功 | 计分 |
|---|---:|---:|---:|---:|---|
| page16 | 1 | 1398.599016 | 13356.166145 | 64 | 否 |
| page16 | 2 | 4069.068385 | 12996.129831 | 64 | 是 |
| page16 | 3 | 4065.841146 | 13028.142156 | 64 | 是 |
| page32 | 1 | 1321.711999 | 20244.962414 | 64 | 否 |
| page32 | 2 | 4072.657398 | 12719.692024 | 64 | 是 |
| page32 | 3 | 4071.892532 | 12705.813262 | 64 | 是 |

两组同为B64S8、guardedKV16/warp1、nativePrefill M256/KV128/w16/s1/L0、budget2048、threshold0、75点预热与gather，唯一server CLI差别是block-size16/32。实际page geometry、原生Prefill执行及扩展supported判断串接NativePrefillLoader均核验；不能只以页长参数作为路由证据。producer/reducer配对及B24回退核验。

384请求逐一重算usage4096/1024/5120、种子、输出、TTFT、耗时及吞吐；双方173/192对输出相同，7项greedy canary相同。不声称输出全部一致，也不替代随机采样Math。9份冻结源码和双方12份安装快照、CLI与75次预热通过独立本地核验。虽然此前Prefill组件更快，此次没有明确服务收益；未采集剖析数据来定量分解各组件贡献。

完整原包3,435,989bytes/14片，SHA `21955de1a27c4345b1d4baf7866c21fa8c2360755791328a363ad032751f9202`。分片与全包SHA均核验，原件在工作区service-results/20261014_tianshu/kv_page16vs32_S8_service。复核：

```bash
python3 experiments/tianshu_kv_page_20261014/verify_downloaded_service.py --root /path/to/20261014_kv_page16vs32_S8_service_tianshu
```

后续K/V planes与direct Decode是独立组件实验。没有16K、新质量结论、安装源修改或沐曦修改。
