# 天数32g：M2048投影分块小规模吞吐提升0.83%，暂不采用

同期新控制与候选各3轮，4K/1K、C64/N64，预先排除首轮。计分均值 **4068.554849 → 4102.387925 total tok/s，+0.831575%**；双方计分轮次不重叠，但收益低于当前2%门槛，不追加独立五轮、新Math或正式benchmark，不更新最好3826.24标签。

候选只路由数值筛选通过的M2048三个BF16 NT投影：O/down BM256/BN256/BK32/w16/s2/group1，gate_up同tile/group8。实际模型中三个shape与配置全部记录并核验。QKV、其他M、dtype/layout/device不符时仍走原路径；不复制重排权重、无量化、无额外输出拷贝。

两组共同B64S8、当前KV16/w1、Native Prefill L0、budget2048/threshold0、75点完整预热和gather、page16、前缀缓存。两组均开启先前已验证的单表SQL反射，因此双方首轮仅约81秒；不能与旧234秒冷轮比较后宣称本GEMM方案的大幅加速，也未复用旧控制。

| 模式 | 轮次 | Total tok/s | TTFT ms | 成功 | 计分 |
|---|---:|---:|---:|---:|---|
| control | 1 | 4041.664924 | 13324.439238 | 64 | 否 |
| control | 2 | 4068.486184 | 13032.380153 | 64 | 是 |
| control | 3 | 4068.623514 | 13001.138231 | 64 | 是 |
| candidate | 1 | 4078.768233 | 12928.916419 | 64 | 否 |
| candidate | 2 | 4103.719902 | 12682.830928 | 64 | 是 |
| candidate | 3 | 4101.055948 | 12687.438642 | 64 | 是 |

各轮新增调优记录数（控制/候选）：`[[0, 0, 0], [0, 0, 0]]`。

384份完整响应逐一重算usage4096/1024/5120、种子、TTFT、耗时和吞吐；11份冻结源码、双方12份安装快照、真实三个GEMM路由、B64S8与B24回退、producer/reducer及75次预热全部独立本地复核。163/192对生成文本相同，7个greedy canary相同；不声称所有生成文本一致或替代随机采样Math。

原包3,568,004bytes/14片，SHA `35e7a5b72e41a1716eff3756949f01db40fbbad2ecb93d55b0aedf15ba880a0b`，逐片与整包校验。原件`service-results/20261014_tianshu/prefill_gemm_tiles_S8_service/20261014_prefill_gemm_tiles_S8_service_tianshu`。复核器`verify_downloaded_service.py --root /path/to/20261014_prefill_gemm_tiles_S8_service_tianshu`。

无新正式/16K或Math结果。此次模型结果说明局部10%算子改善最终约0.83%，继续优先检验框架预算等整体路径。安装源、官方benchmark与沐曦仓库不变。
