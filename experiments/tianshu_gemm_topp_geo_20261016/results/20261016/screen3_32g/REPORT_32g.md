# 天数：几何TopP与M2048 GEMM组合4K短测提升2.25%

运行源e84bad54184fc36248dd2656205d15f92f21a2dd，之后ccfe897仅新增未启动CPU诊断；56份运行源码实际全部不变。原始官方serve argv，默认2048预算/256序列/async=True/75暖点、CV/KV16/S8、nativePrefill L0、原生采样RNG。候选仅M2048三个GEMM配置和TopP32K global-max归一化、概率域重复精确判断与几何中点搜索；TopK和B1..7回退，采样p、温度及RNG不变。

|模式|轮|total tok/s|output tok/s|TTFT ms|打印调优记录|
|---|---:|---:|---:|---:|---:|
|control|1（排除）|3743.289089|748.657818|16776.960498|0|
|control|2|3929.248502|785.849700|13198.930787|0|
|control|3|3950.931935|790.186387|13185.719269|0|
|candidate|1（排除）|3387.700881|677.540176|26639.450339|0|
|candidate|2|4025.689825|805.137965|12755.805522|0|
|candidate|3|4031.994666|806.398933|12780.727329|0|

N64/C64、4K输入/1K输出，3+3轮去首轮、输入逐轮换seed且两模式配对。后两轮均值3940.090219→4028.842246 total tok/s，+2.252538%；TTFT 13192.325028→12768.266426ms（-3.214434%）。控制/候选轮间差距0.550328%/0.156493%，且候选两轮均高于控制两轮。

达到预设至少2%、去首轮区间不重叠及各自差距小于1%的全新质量验证门槛。全新105题Math于2026-10-16 17:42开始，官方temperature1/top_p.95/max_tokens32768，硬截止17:58。本报告发布时未完成，不复用旧103/105，不提供本版完整正确率或正式成绩。正式最佳仍3992.97 total tok/s、798.60 output tok/s、TTFT4443.93ms，原版本Math103/105。

384请求文本/usage/seed/计时、4个C1 canary（仅回退路径）、16份零preemption metrics、实际38环境/原始argv、56源码快照/当前仓库源、B64实际geo源09156e66及M2048三项配置均下载复核。CPU诊断没有运行或混入吞吐测试。0条打印运行期autotune记录，不证明无未打印JIT；候选首轮额外代价保留，未隔离根因。

原件 ../service-results/20261016_tianshu/gemm_topp_geo_service/20261016_gemm_topp_geo_service_tianshu，6040479bytes/24parts，SHA256 `5d6196fe8405e6ebff756c05e012ca7f7415f42eaca74048c4015cc788d3d5ab`。分片和全包均核对；首次两分片重试后成功，没有以观测超时判作任务停止。runner45057及控制/候选服务均终态；Math是另一个独立的新服务。

```bash
python3 experiments/tianshu_gemm_topp_geo_20261016/verify_downloaded.py \
  --root ../service-results/20261016_tianshu/gemm_topp_geo_service/20261016_gemm_topp_geo_service_tianshu
```
