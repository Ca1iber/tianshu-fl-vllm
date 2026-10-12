# Spec接入既有native Prefill：短协议有明显收益

保持4K输入/128输出、C64/N128、target图/draft eager、三轮排首轮，固定同一DSpark模型及7步概率采样，M128/KV64 target验证tile不变。新增target causal、平均query≥16的2D批次使用已补查6场景的M256/KV128/w16/s1，draft非因果及3D decode不进入新tile。不是官方4K/1K/N256协议。

仅验证tile的Spec均值3200.421505→加入Prefill的8133.276910 tok/s，提升154.13%左右，TTFT约23005.20ms。候选三轮1967.504180/8131.938601/8134.615220；冷轮274.800941s，三轮均无新增autotune记录，但冷轮仍包含其它首次执行/编译成本，明确排除首轮。384计时请求成功，usage全为4096/128。保留原始响应、逐轮metrics和冻结源码。

结论是将已有Prefill优化接入Spec的收益，不将全部增幅归于投机算法；当前非投机最佳同样有Prefill优化，还需用匹配随机输入的4K/1K/C64/N128对照确认整体优势，再独立Math-500，才考虑正式benchmark。

完整备份13065651 bytes、SHA9da136f8a0a06dc223be1a9943d77a8ebfebf0a479324999ffc08908cfe75348已本地逐块及整包核验。尚未质量/正式采用，最佳tag仍为aa8fac4对应2712.55。
