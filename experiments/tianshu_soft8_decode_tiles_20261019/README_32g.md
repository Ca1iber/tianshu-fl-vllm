# 软件8位KV decode分段与tile筛选

临时BF16 prefill已经解决长prefill严重退化，但旧byte decode在B16/B24和mixed63仍慢。独立扫描S4/8/16/32/64、T16、w1/2，七配置；BF16最佳与原Soft8 dispatch均保留为对照。十场景覆盖B1/3/8/16/24/32/63/64、4K/5K/短33、strided cache/Q/out。保持codec/writer和attention算法，单独调整分段与launch参数。

每个配置实际builder/writer/forward，两个变Q/K/V/长度/slots的Graph重放、完整物理writer bitwise、output与segment scratch尾部、RNG、量化后FP32/CPU参考。各7组×20回放包括writer，记录实际compiled资源和参数。配置之间使用独立scratch，防止不同segment活跃区污染边界断言。总630事件组；没有模型服务或Math证明。
