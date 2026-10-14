# 天数BI-V150 32g：Decode .cv缓存策略独立五轮确认

双边重新启动、4K/1K/C64/N64、temperature1/top_p.95，全部保持4096预算/83点/S8、NativePrefill、SQL、原生TopP。仅支持的3D Decode KV load缓存提示从默认改.cv。预先排除前两轮，后3轮计分；不复用三轮筛选的控制分数。

|组别|第1轮（排除）|第2轮（排除）|第3轮|第4轮|第5轮|后3轮平均 tok/s|
|---|---:|---:|---:|---:|---:|---:|
|默认KV缓存|3722.01|3976.74|3971.52|4000.82|3981.77|3984.70|
|.cv|3783.04|4060.64|4044.33|4079.26|4080.12|4067.90|

提升 **2.0880%**，计分候选最慢4044.33仍高于计分对照最快4000.82，通过预先设定>=2%且三轮无重叠门槛。候选后3轮约0.88%波动，不能称为零波动。

640请求全部usage、种子、非空文本、TTFT和耗时重算通过；16冻结源码、实际83点、安装6源、64种batch路由的真实.cv/源SHA8deb、producer/reducer匹配与完整生成源来源核验通过。新的独立Math已由串行控制器实际启动，未复用旧103/105，正式成绩仍未更新。

整个候选阶段无新的请求期调优记录；这按各轮原始日志边界核对，不把缺少GEMM调优等同于任何阶段完全无调优。当前改变是cache load提示，不涉及计算精度、量化、投机解码或输出长度。

归档5888723bytes/23parts，SHA256 `6e4264d7622da67dd8ba8d6a258529840a9d91bf950717b2429d8304551f25fa`。本机raw `service-results/20261014_tianshu/decode_cache_cv_confirm5/20261014_decode_cache_cv_confirm5_tianshu`。复核：`python3 experiments/tianshu_decode_cache_service_20261014/verify_downloaded_confirm.py --root <raw-root>`。

后续新Math105严格>95%后，运行原官方4K N256四轮skip1。完整下载并核对质量/正式结果前，不采用为新best，当前仍4000.99 total tok/s。
