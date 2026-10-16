# 天数 BI-V150 32g：4K负载的调度预算4096/16384对照

新鲜同条件4K输入/1K输出、C64/N64，每组3轮，预先排除首轮。16384只指每步调度token预算，本轮没有运行16K输入benchmark。两组均当前CV/L0/KV16/w1/S8、native Prefill、原生TopP、max_num_seqs256、async=True；保持75点基础规则，control实际83预热点，candidate实际131点，额外大M的GEMM调优安排在启动。

|组别|轮次|total tok/s|output tok/s|TTFT ms|
|---|---:|---:|---:|---:|
|4096预算|1（排除）|3793.84|758.77|17582.23|
|4096预算|2|4064.10|812.82|11739.13|
|4096预算|3|4055.47|811.09|11747.86|
|16384预算|1（排除）|3090.42|618.08|29874.30|
|16384预算|2|4086.40|817.28|12131.71|
|16384预算|3|4103.97|820.79|12120.56|

稳态均值4059.7851→4095.1872 total tok/s，提升 **0.872020%**，TTFT约高3.26%。不单独采用，不因此运行完整benchmark或新Math；作为后续组合方向保留。不能将N64短测4095.19直接替代正式N256最佳4091.29。

两组384响应全部成功并本地核验输入生成种子、4096/1024用量、正文、TTFT/墙钟及聚合吞吐。18份冻结源码/快照、6安装源、Worker真实预算/最大序列数、83/131实际预热、真实131点helper SHA和单个守卫常量改变、CV生成源码及KV16/S8 producer/reducer均复核。KV容量control521024、candidate513360tokens，均>=327680；12份原始metrics的preemption均为0。

候选首轮记录3次_index_linearized_jit_function调优，打印的完成时间合计0.73秒，后两轮无新增调优；不能用这些打印时间解释整个首轮差距。启动131点预热不等于所有非GEMM路径已经预热。本轮没有新Math，不借用旧103/105证明候选质量；正式最佳及对应质量保持。

归档5302977bytes/21parts，SHA256 `03d0bee712dd272da219b53d0785e3051aa2cf49f9af8c9df21f290c14512f6a`。本机raw：`service-results/20261016_tianshu/budget16384_service/20261016_budget16384_service_tianshu`。

复核：`python3 experiments/tianshu_budget16384_20261016/verify_downloaded.py --root <raw-root>`。当前预算实验已终态并释放GPU；下一独立async调度开关实验已启动，继续沿4096/83的当前最佳配置。
