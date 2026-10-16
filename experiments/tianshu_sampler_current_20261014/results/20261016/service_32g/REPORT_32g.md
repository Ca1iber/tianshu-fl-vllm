# 天数 BI-V150 32g：当前最佳配置下的采样融合复测

本轮为新鲜同条件4K输入/1K输出、C64/N64服务对照，每组重启后3轮，预先排除首轮。两组均CV、L0、KV16/w1、B64 S8、budget4096/83点、native Prefill、nativeTopP w16/s1、max_num_seqs256；候选只将过滤后的softmax/div/argmax换为此前检查的指数竞赛融合，过滤与全局/每请求指数噪声生成沿原路径。

|组别|轮次|total tok/s|output tok/s|TTFT ms|
|---|---:|---:|---:|---:|
|当前配置|1（排除）|3800.04|760.01|17452.78|
|当前配置|2|4063.54|812.71|11730.73|
|当前配置|3|4054.86|810.97|11762.02|
|融合采样|1（排除）|3800.83|760.17|17444.78|
|融合采样|2|4072.69|814.54|11752.45|
|融合采样|3|4072.47|814.49|11796.35|

后两轮均值4059.1974→4072.5769 total tok/s，提升 **0.329608%**。没有明显端到端收益，不采用，不因此追加完整benchmark或新Math。不能将该N64短测与正式N256最佳4091.29直接当作同条件对照；正式最佳及Math103/105不变。

两组384条吞吐响应全部成功；输入生成种子、4096/1024用量、非空文本、逐请求TTFT/墙钟、聚合吞吐均本地复算。18份冻结源码和快照、安装源快照、实际Worker max_num_seqs256、两边83点预热、CV生成源码、KV16/S8 producer/reducer、原生TopP配置均核验。候选实际sampler kernel SHA为 `83739846d48d6e12710d9cdbb0da61debb733241e8aa45e1ea1317bdd2950e93`；记录覆盖B64（总65种batch首次路由，包含启动阶段，次数不代表耗时占比）。两组记录的运行期autotune列表均为空。

旧105例独立固定噪声检查针对同一kernel，不能替代本候选新Math；本轮未跑新Math，不声明所有浮点输入逐位相同，也不以随机输出文本相同证明正确率。

归档5322575 bytes / 21 parts，SHA256 `ee37eaa073d645845d4d9a1a41fcb73ae860ed82a1bb8e4aa2218ef99e80c49c`。本机raw：`service-results/20261016_tianshu/sampler_current_service/20261016_sampler_current_service_tianshu`。

复核：`python3 experiments/tianshu_sampler_current_20261014/verify_downloaded.py --root <raw-root>`。下一项保留4K输入，仅增大调度token预算至16384，带131点预热，独立新鲜对照并核验KV容量及零preemption。
