# DSpark eager C8短对照

固定相同target参数：BF16、budget2048、maxlen131072、gmem0.85、maxseq8、eager，native无sink L0修正，原tile，Hybrid/adaptive关闭。候选额外使用5层DSpark draft、固定7步顺序Markov概率采样及原始target rejection sampler。请求为4K随机token输入、128输出、ignore_eos、temperature1/top_p0.95，C8/N16，3轮排首轮；同seed生成的请求负载一致。不是官方4K/1K/C64/N256协议，也不是当前最佳图模式组合。

非投机三轮579.253554/1596.193647/1598.284837 total tok/s；候选97.919505/2969.935320/2968.277766。排首轮均值1597.239242→2969.106543（+85.889907%）。TTFT4139.364940→4614.768525ms（+11.484940%）。两版本各48请求成功，每个请求usage完整且输入4096/输出128；后两轮均无新增调优。两边后两轮吞吐波动分别约0.13%与0.056%。

候选冷轮690.199568s、84项调优记录；非投机冷轮116.674295s、26项调优。明确排除首轮，但完整保留这些启动/首次请求成本。尚需解决预热覆盖，不能只公布快的稳态而隐去冷成本。捕获日志中的调优时间不涵盖全部编译开销。

完整原始响应、log、metrics、source snapshot及source bundle全部本地SHA验证。非投机包3370787 bytes/SHA6444ba8c5e8069eac991b0ec18fe83c0aa8312f4641ae61b3dbf3e2442d4a2a8；候选包3370932 bytes/SHA72caa4f65606d12d5309c03abfb88779dad507025ec70527e61cdf28b944d0cd。尚未独立Math-500或正式benchmark，不更新2712.55正式最佳。后续先验证C64及target图模式，再检查验证注意力tile、采样视图、预热，独立质量合格且有稳健收益后才运行官方4K。
