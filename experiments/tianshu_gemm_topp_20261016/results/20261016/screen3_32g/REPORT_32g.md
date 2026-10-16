# 天数：原始命令下M2048 GEMM+strictTopP16K组合提升1.19%，暂不采用

本次保持官方原始serve argv，默认budget2048/maxseq256/asyncTrue/stream1、75实际预热点、CV/KV16-w1/B64S8/nativePrefill/原采样器；候选只开启三个M2048 GEMM tile和strictTopP16K。两组重新启动服务，各3轮4K/1K/C64/N64，输入seed逐轮不同且跨组配对，预先排除第一轮。

|组别|轮次|total tok/s|TTFT ms|计分|
|---|---:|---:|---:|---|
|control|1|3740.749180|16655.883247|否|
|control|2|3938.711373|13143.608679|是|
|control|3|3939.547688|13163.662055|是|
|candidate|1|3810.015545|16327.070183|否|
|candidate|2|3981.220771|12774.379965|是|
|candidate|3|3990.914946|12749.853894|是|

计分均值3939.129531→3986.067859，+1.191591%；同期TTFT下降2.976504%。虽计分轮无重叠，仍低于事前2%门槛，不进入独立5+5、新Math或正式测试。prepared脚本保留但未启动，未降低门槛。当前正式4K3992.97 total tok/s与Math103/105不变；N64筛选值不能直接与N256正式值比大小。

384响应全部成功，正文、4096/1024/5120 usage、输入seed、TTFT、耗时和吞吐重算通过；4个C1 seed777 canary文本一致只证明回退路径，不代表B64新Math。52份运行源码快照与当前仓库对应源一致，实际38环境仅两个开关不同，原始命令/default2048/256/75warm/CV/S8、三种M2048配置、B64严格TopP源029d以及16份metrics零preemption均独立本机复核。六轮请求期新增调优打印均0，不据此宣称所有编译成本都已覆盖。

M2048路由首次记录可能含启动dummy/warmup，TopP同样每batch/route只记录首次；这些证明实际函数与配置，不能当成逐请求计时阶段调用计数。端到端收益依据完整客户端响应时长。旧M2048和TopP服务条件不同，不能相加其历史收益。候选采样RNG保持原实现，TopP修复及GEMM分块仍可改变输出，未借旧质量。

运行源准备90acb588679cfb567644a7f12aeeb43ce8807d57；462b4b5只新增未执行的确认/验证脚本，部署前后52运行源逐一SHA未变。parent36242、control36243/API36245、candidate37138/API37140/Engine37295均终态，ixsmi已确认GPU空闲。用户WIP保留，未推送、未改沐曦。

原始数据位于service-results/20261016_tianshu/gemm_topp_service/20261016_gemm_topp_service_tianshu。5930009bytes/23parts，SHA256 `c5b12cd794e1c80715a366cf8e7dfe2a724264b5336404a2c2644e594a795914`；全部分片及整包校验通过。实际执行以下命令通过：

```bash
python3 experiments/tianshu_gemm_topp_20261016/verify_downloaded.py \
  --root ../service-results/20261016_tianshu/gemm_topp_service/20261016_gemm_topp_service_tianshu
```
