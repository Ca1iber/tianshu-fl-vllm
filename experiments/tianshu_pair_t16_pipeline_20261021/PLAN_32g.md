# 现行成对K读取器的T16 warp/stage补充扫描

前次pair_tile_sweep已证明T32/T64负收益，但未测T16的更多warp/流水线。更早10-19 c6bbc4c7旧K8V16约142regs也测过warp和stage2且负收益；本轮针对现行f1df731b、176regs/0spill的成对读取，不把旧代码结果当相同实验，也不预言收益。

七路current w1/s1、w2/s1、w4/s1、w1/s2、w2/s2、w1/s3、w2/s3；只n>24改变，全部T16、原分段/容量/.cv/12KiB K8V16和真实writer不改。n1..24各路维持现行配置。旧native Producer会固定stage1，本轮在真正JIT调用前单独设置stage，保留原Producer/Reducer源码不动，CPU stub与实际GPU metadata都必须证明最终参数和二进制hash。

沿用14场景，两阶段同Graph变更Q/K/V/length/table/slot；包括C64 4K/5K、ragged/strided、40/16K单层组件及length1捕获后4K/8193重放。预计196完整GPU输出、304独立CPU参考向量、2128已记录CPU检查；九轮正反交替×20热Graph，882计时组，包含真实writer+attention+reducer。原.002+.01误差门槛不放宽，bitwise另记，保留全部慢配置与编译/数值失败。

CPU实际token覆盖4928组、捕获到长序列容量3080组，另外actual-launch stub确保stage未被旧层覆盖。绑定现行407源、原正式B64 S4/T16/w1/stage1证据，不只用相同文件名认定相同代码。GPU前冻结源/CPU gate/计划，真修复另目录。独立CPU复核结束后才判是否值得服务N64×3与新Math>95，之后才正式N256；当前推荐5586.28不变，不常规16K。

GPU23:58北京硬停；代码/报告/关键JSONCSV只容器+GitHub，原始PT/编译产物/完整日志仅容器，不动用户两份dirty、FlagGems或沐曦仓库，无子agent。
