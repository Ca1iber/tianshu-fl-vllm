# 天数 BI-V150 32G：K8V16预算4096/8192短测计划

Local screen, not a formal score. 两侧相同K8V16与最佳算子，显式CLI token预算4096→8192，预热83→99。保留基础75点、maxseq256、async=True、采样算法与缓存格式。旧BF16下8192只比4096快0.6088%，不将旧结论外推到KV流量已减少的新配置。

3+3 fresh服务，各预排首轮，C64/N64、4K/1K，完整384响应/41环境/源快照、真实worker预算/实际warm与零抢占审核。>=2%且稳定才追加新Math，不借其他profile正确率；不运行16K。CPU只检查真实argv转换，未测GPU/Math或硬件profile。

优先级在当前预算4096的新Math与正式4K之后；有空余GPU时间才启动。本候选不提前采用，原命令正式best4997.05保持。
