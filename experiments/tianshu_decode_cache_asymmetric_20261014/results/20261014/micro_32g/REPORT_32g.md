# 天数 BI-V150 32g：Decode K/V 分别缓存与 eviction 策略

保持当前 .cv、L0、KV16/w1、B64 S8 配置，仅将 K/V 的加载提示分开。9 种策略包括当前策略恒等对照、K 或 V 单独恢复默认或 .cg，以及四种 evict_first/evict_last 组合。算术、producer/reducer 网格及分段不变；生成源码逐字反向替换可还原当前 .cv 源码。

17 个场景、153 个候选场景、1530 个原始 event 中位数均已本地重算；所有候选原输入与变更 Q/K/V 后 Graph 重放均与当前实现逐位相同，FP32/选定 CPU 参考及尾部保护检查通过。6 份冻结运行源码、实际生成源码、编译器快照与 10 组真实 kernel 产物 SHA 均核验。此处没有运行服务或 Math，不外推端到端收益。

|场景|当前耗时 ms|该场景最佳变动|
|---|---:|---:|
|B8 KV4096|0.140430|-3.2576%（K .cg / V .cv）|
|B16 KV4096|0.263647|-0.1326%|
|B24 KV4096|0.412454|-0.9808%（恒等对照）|
|B32 KV4096|0.503418|-0.1251%|
|B64 KV4096|0.845941|-0.0380%|
|B64 KV5120|1.053130|-0.1177%|

C64 主负载没有实际收益：K 或 V 单独恢复默认/.cg 反而慢约 2%～3%，eviction 组合的变化不足 0.12%。B8 的局部改进不能推为 C64 吞吐收益，B24 恒等对照的变化也表明应避免解读小幅差异。保持正式最佳 4091.29 total tok/s、Math 103/105，不采用本实验。

原始归档 4284421 bytes / 17 parts，SHA256 `002786febbe00d81a90ee3ad5f88c66e2416befe090c6faae2299cded678a2a7`。本机 `service-results/20261014_tianshu/decode_cache_asymmetric/20261014_decode_cache_asymmetric_tianshu`。

复核：`python3 experiments/tianshu_decode_cache_asymmetric_20261014/verify_downloaded.py --root <raw-root>/micro`。
