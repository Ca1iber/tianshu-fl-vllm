# 天数 BI-V150 32g：int32 attention 正式16K结果

原始16K/1K、C64/N128、四轮排除首轮：2600.48 total tok/s、152.97 output tok/s、TTFT189964.79ms。对比旧正式16K 2440.62，吞吐提升6.549975%，TTFT降低5.481775%；对比公开915.15，吞吐提升184.158881%。

|轮次|total tok/s|output tok/s|TTFT ms|duration s|纳入均值|
|---:|---:|---:|---:|---:|---|
|1|2358.03|138.71|198807.95|944.95|否|
|2|2601.61|153.04|189858.18|856.48|是|
|3|2600.93|153.0|189914.32|856.7|是|
|4|2598.91|152.88|190121.87|857.37|是|

四轮512/512成功，各128请求，usage总数逐轮2097152输入/131072输出，0失败。后3轮极差/均值0.103827%。新地址32配置独立Math102/105、97.142857%、0执行errors，与本次hook/helper/生成内核hash完全匹配；没有复用旧profile质量分数。

保留Native L0 Prefill M256/KV128/w16/s1、adaptive Decode KV16/w4/s1、budget2048、75 Prefill buckets和fixed logits gather。新增仅Decode安全守卫下的int32元素偏移/stride，B4..48、BF16/NONE、所有带余量元素offset<2^31；浮点计算和物理指针保留。未合入独立SQL反射或KV写入启动配置实验。旧自动审计模板只描述stage1基础profile，这份报告和comparison显式补充actual address32源码/路由，避免误读为仅stage变化。

全部原始日志、四轮响应/CSV、真实源快照和质量匹配审计归档2291386 bytes、9分片；整包SHA256 015ea50aba9c6d186952fec1e355bfab7be9e8a3fc07c7687702cc41ac9a67e4，所有分片及整包本地verified。原始目录service-results/20261013_tianshu/attention_address32_official16k/official16384。Git CSV复制统一LF，原始归档保持原字节。Benchmark源码SHA0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565未改。

本次正式16K已验证有收益。正式4K仍在顺序测试，完整profile最终选择及canonical best更新等两个工作负载均闭合后再进行，暂不覆盖原可回退配置。
