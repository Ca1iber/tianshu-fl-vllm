# dotsaddz W4A16 Down64：图执行+19.6068%，尚未模型集成

同一r3 BF16模块/环境对照，真实42层Down64不同bank轮换图：2.320544910 → 1.940143871 ms，+19.6068469%，绝对节省0.380401039 ms/解码步。eager：3.761924934 → 2.035523605 ms，+84.8136%；eager包含CPU调度差异，Down实际以图执行为主，主要看图结果。每模式7轮、每轮10次全42层，交替顺序，排除首轮后取6轮中位数。

2359296格完整FP64校验全部通过；42层三phase的全部BF16输出图poison/eager一致、有限。完整输入/原权重/pack/scale/zero字节、版本、边界、RNG、903来源和checkpoint都保持不变。量化对原BF16最大投影相对L2仍为13.8703314%。采用group128 RTN、先BF16 scale后round/clamp、自然N打包、Z=-8*S的验证dotsaddz格式；不是经过激活感知校准的AWQ模型。

旧AWQ的图组件+6.0149%仅为另一轮运行参考；本版直接配对的是BF16与dotsaddz，不混用跨运行时间当paired结果。组件百分比不等于整机收益，尚未运行Math500或正式benchmark。输入来自真实GateUp权重生成的旧fixture，不是在线模型activation。模型架构和checkpoint未更改，服务未采用本版。

源码冻结5c79e84924b33eaa97148db8adc80871ecc77c91；容器原始目录 `/workspace/benchmark-results/20261023_native_dotsaddz_down42_{CPU,GPU}_tianshu`。GitHub保留源码、关键计时、CPU门禁和结果SHA；大数组仅容器。正式最佳仍为r3 5950.75 total /1190.15 output tok/s。下一步完成真实LM Head与当前已启用W8A8的组件对比，再根据绝对节省选择模型候选。
