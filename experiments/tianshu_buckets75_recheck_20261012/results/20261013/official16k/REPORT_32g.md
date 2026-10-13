# 天数 BI-V150 32g：完整16K正式结果

同已验证4K组合：Hybrid、adaptive Decode KV16/w4/stage1、Native GQA Prefill M256/KV128/w16/stage1（L0）、budget2048、75点Prefill buckets、固定最后token gather。没有投机解码或整数计数修改；模型、BF16权重、原benchmark逻辑保持原样。

原负载16384输入/1024输出、并发64、每轮128请求，完整四轮排首：**2440.62 total tok/s、143.57 output tok/s、Mean TTFT200982.18ms（200.98218秒）**。四轮512/512成功，每轮输入2097152、输出131072token。

|轮次|总tok/s|输出tok/s|Mean TTFT/ms|耗时/s|计入|
|---|---:|---:|---:|---:|---|
|1|2208.7|129.92|210990.67|1008.84|否|
|2|2442.69|143.69|200801.55|912.2|是|
|3|2440.16|143.54|201028.81|913.15|是|
|4|2439.02|143.47|201116.19|913.57|是|

后三轮极差/均值0.150371%，平均每轮912.97秒。四轮负载合计3747.76秒（不含服务启动、CLI加载和归档）。

相对用户提供的官方16K公开基线915.15 total tok/s、53.83 output tok/s、TTFT599262.03ms：吞吐提升166.690706%（约2.67倍），TTFT降低66.461720%。这里的915.15是公开参考，不是我们此前优化版完整最好。此前优化组合没有完成正式16K对照，不能量化此版本相对旧优化组合的16K净增益。

同配置独立Math-500 Level3为102/105=97.142857%、零错误；与本次运行的worker/adaptive/hybrid、native kernel、Decode hook、bucket/gather源SHA及实际路由全部匹配。完整4K已验证3222.52 total tok/s、644.51 output tok/s、TTFT4510.23ms；两种负载分别保存，不能把16K total与4K output混比。

运行监控累计16次抢占，每轮结束约累计4/8/12/16次，未发生请求错误。实际KV容量522816token，运行约30请求、其余排队。这表明即便吞吐稳定，仍有容量调度改进空间；后续独立测试watermark .035/.07，当前命令没有添加watermark。未采集硬件profiler，不能把抢占计数等同于全部性能瓶颈。

原始归档2159196bytes、9分块，SHA256 `7302e7263de34e079e607c8ecf15262987e3baefafcd28219a086c6d5dac3d8e`，全量本地下载核验。原benchmark SHA256 `0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565`。Git内CSV仅CRLF规范为LF，原始字节保留在归档。

复现启动器：`experiments/tianshu_buckets75_recheck_20261012/serve_tianshu_minicpm.sh`；原benchmark testcase `[[16384,1024,64,128]]`，默认四轮、跳过首轮。

结论：本组合完整4K、16K和独立准确率均闭合，16K后三轮较稳定。接续整数计数GPU验证，再测实际服务收益；沐曦仓库保持只读。
