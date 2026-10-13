# 天数 BI-V150 32g：L0 DecodeKV32同期4K服务收益

4K输入/1K输出、C64/N64、每组五轮，预先排除前两轮：同期控制3246.262382→候选3467.708514 total tok/s，提升6.821572%。两组后三轮没有重叠：候选最低3451.69高于控制最高3261.80。TTFT13508.220541→13355.380299ms，下降约1.13%。这是缩小请求数的小测，不能替代官方N256成绩。

|轮次|控制total tok/s|候选total tok/s|控制TTFT ms|候选TTFT ms|计分|
|---:|---:|---:|---:|---:|---|
|1|1266.751374|1299.185907|19259.428648|19756.887914|否|
|2|3259.137063|3458.882236|13498.978083|13344.536602|否|
|3|3230.200357|3483.897849|13513.758462|13346.335351|是|
|4|3246.789624|3451.685318|13507.684026|13342.733489|是|
|5|3261.797167|3467.542375|13503.219135|13377.072056|是|

控制后三轮波动0.973329%，候选0.928928%。640个吞吐请求全部成功，逐轮提示种子一致，每项4096输入/1024输出。7项greedy检查只有6项文本相同；2048-token随机提示的16输出tokens出现差异，不声明逐位或所有greedy等价。新配置独立Math正在运行，旧Math成绩不复用；通过官方>95%门槛才跑正式4K，目前不升级最好成绩。

候选仅将Decode改为无sink L初值0、KV32/w4/stage1，producer/reducer的TILE都为32；源body的其他部分及网格、64位整数偏移/物理指针保留。Native Prefill L0/stage1、adaptive segments、75桶/固定gather、budget2048、BF16等保持控制配置。没有混入此前Decode32、Prefill stage2、SQL反射、KV writer、投机或量化修改。

先决数值证明为22长短场景、156 CPU参考及GPU FP32/tail/Graph检查通过，见上级micro_32g报告。服务最终64条producer和64条reducer记录逐项配对，producer实际KV32/w4/stage1、reducer实际TILE32、生成源SHA4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e与数值证明一致。原native源SHA bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0。fallback路径保留。

完整原始归档4755234bytes、19分片，SHA256 77837a4b69f873dc5c9ab5a979e70a5da3d758423a079219da6388f254587abb；分片/整包、运行source_snapshot、640请求usage和种子、计分均值及两阶段路由已本地核验。目录service-results/20261013_tianshu/decode_coherent_service/20261013_decode_coherent_service_tianshu。安装源、权重和官方benchmark未改，沐曦只读。

结论：候选有明确小规模4K服务收益，继续独立质量和正式4K验证。硬件性能剖析未采集，TODO。后续只安排4K正式负载，没有新增16K服务benchmark。
