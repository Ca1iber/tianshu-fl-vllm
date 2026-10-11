# 天数FA2 Prefill tile扫描（服务验证中）

42个组合完成固定FP32 atol0.02/rtol0.01检查，24个选定CPU参考检查通过，padded tail保留；没有数值失败。六个真实paged/interleaved/GQA16/2/D128场景均以BM128/BN128、8 warps、2 stages最优，算子耗时下降约29.5%–42.3%。BM64的一些候选更慢，全部失败/负例及原始计时都保留。当前仅为单层Attention测量，不能称为服务或正式吞吐收益。

控制为原BM128/BN32、4 warps、3 stages。只在独立实验进程临时覆盖mha_block_128的heuristic，其他key不改，未写site-packages。已在预算2048、Hybrid1/adaptive1、buckets0/gather0下启动4K输入/128输出、64并发/192请求、三轮排除首次的默认/候选服务对照，runner68624。实际tile记录可核验。新路径独立数学和正式成绩尚未完成，不能复用旧准确率。

完整数值/CPU/全部计时备份SHAf6c5c752fce8a078f93a0a3e9d62d24ae93bdbfbfeaef4f6affa8aee7e0138d3，16979 bytes，已本地校验提取。当前GPU仅运行该服务对照。后续按数据决定是否扩到4K/1K以及独立质量评测，11:58停止GPU工作。
