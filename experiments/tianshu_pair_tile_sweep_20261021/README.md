# 成对K读取后的tile/warp独立扫描

对照为当前paired-K原源码f1df731b，T16/warp1；n>24时比较T32/warp1、T32/warp2、T64/warp2，n1..24完全保持。不是重复此前短上下文S2/S8/S16扫描：分段策略与容量不变，K8/V BF16格式、writer和reader每一个字节都不改，只改变已合格启动器允许的tile与warp参数，producer/reducer采用同一tile。

14场景×4路×两次同Graph变更Q/K/V/length/table/slot；包括C64 4K/5K、ragged/strided以及长度1捕获后到4096/8193重放。保留全部FP32/CPU输出参考、完整物理缓存、输出与scratch边界、RNG与动态输入检查。改tile会改变浮点归约顺序，bitwise另列诊断，不放宽原有.002+.01*abs(reference)门槛。

CPU遍历2816个完整token一次覆盖、1760个捕获到重放容量场景，含实际tile的不同chunk长度。九轮交替×20热Graph回放，计时含一层真实KV写入与decode，不是模型或正式吞吐。只有有足够、可复查局部收益才考虑服务A/B和新Math；不例行跑完整16K。

原件只留容器，GitHub保留源码、报告与关键JSON/CSV。串行等待当前CPU诊断终态及GPU空闲；所有GPU受2026-10-21 23:58北京时间截止约束。
