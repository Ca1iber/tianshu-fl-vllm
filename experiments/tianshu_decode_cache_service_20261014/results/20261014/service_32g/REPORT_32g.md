# 天数BI-V150 32g：Decode K/V .cv缓存提示服务对照

保持当前4096预算/实际83点、KV16/S8、SQL、NativePrefill、原生TopP，temperature1/top_p.95；4K/1K/C64/N64，双方新启动3轮，预先排除首轮。仅支持的3D Decode K/V两处load使用.cv提示；控制用原L0内核，候选用独立生成源。保护条件与整个reducer AST不变，算术、整数宽度、网格、tile与warp策略不变。

|组别|第1轮（排除）|第2轮|第3轮|后两轮平均 tok/s|
|---|---:|---:|---:|---:|
|原缓存策略|3722.17|3973.99|3968.95|3971.47|
|.cv|3681.41|4059.80|4055.21|4057.50|

小规模均值提升 **2.1663%**，候选两轮均高于对照两轮。符合预设独立5+5确认条件，尚不采用为正式最佳。没有用短测4057.50替代官方N256分数。

384请求逐条usage、种子、非空文本、TTFT和耗时重算通过；15冻结源码、双方真实83点、安装6源、KV16/S8及reducer匹配通过。控制实际64种batch路由源SHA4a1f.../无显式cache提示；候选64种batch路由均SHA8deb...、实际.cv。完整生成源与cache生成来源已归档，原始L0路径保留，无虚报源SHA。

CPU准备检查128组producer/reducer与44个不支持配置回退均通过；真实组合hook分派及异常时恢复验证通过。已有单算子probe的51组/612事件及实际IR缓存操作数差异为机制证据，不把它当作本服务的缓存命中率测量。

完整归档5034322bytes/20parts，SHA256 `1556c381bcbf9614efeb48627c1172219bc7f657d51ec4188218f67a13e3f6e6`。本机raw `service-results/20261014_tianshu/decode_cache_cv_service/20261014_decode_cache_cv_service_tianshu`。复核：`python3 experiments/tianshu_decode_cache_service_20261014/verify_downloaded.py --root <raw-root>`。

独立5+5确认已另开新服务、预先排除前两轮，使用640请求。确认若>=2%且计分轮次无重叠，串行执行新Math105严格>95%及原官方4K四轮；未借用旧Math。当前正式最佳4000.99及Math103/105保持。
