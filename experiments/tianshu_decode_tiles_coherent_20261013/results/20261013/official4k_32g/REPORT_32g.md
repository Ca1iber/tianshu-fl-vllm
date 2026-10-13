# 天数 BI-V150 32g：L0 DecodeKV32正式4K最好结果

原始官方4K输入/1K输出、C64/N256、四轮排除首轮：3426.62 total tok/s、685.32 output tok/s、平均TTFT4514.45ms。比此前正式最好3222.52提升6.333553%，比官方公开2028.01提升68.964650%。TTFT相对旧4510.23ms变化+0.093565%，基本持平。

|轮次|total tok/s|output tok/s|TTFT ms|耗时s|计分|
|---:|---:|---:|---:|---:|---|
|1|2152.59|430.52|7114.08|608.90|否|
|2|3313.92|662.78|4520.83|395.52|是|
|3|3465.48|693.10|4493.94|378.22|是|
|4|3500.45|700.09|4528.59|374.44|是|

四轮1024/1024成功，每轮1048576输入、262144输出tokens，0失败。后三轮极差/均值5.443562%，仍有波动；没有丢弃第二轮，没有把最后单轮3500.45或小规模3467.71当作正式均值，未声称完全稳定。

对应新独立Math Level3为103/105（98.095238%），105项唯一且完整、0执行错误，满足严格>95%。正式评测actual producer/reducer共64组配对，TILE都为32，producer w4/stage1、无sink L初值0；生成源SHA4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e。candidate hook、原native源、75桶/gather相关源与Math一致，源快照逐字核验。新独立质量不复用另一套stage2的同分数。

共同配置是Native GQA Prefill M256/KV128/w16/stage1 L0、adaptive Decode KV32/w4/stage1 L0、75点预热桶、固定末token gather、budget2048、BF16。这里KV32表示每个Decode KV tile处理32个tokens，整数元素偏移和64位物理指针未改，不是KV量化位宽。未混入Decode32位地址、Prefill stage2、SQL反射或KV写入实验，未投机、未量化。

原benchmark SHA0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565未修改。正式归档2450682bytes、10分片，SHA256414818244b69a6df17ae5e1803613f0bca3446d76d25a2a5b65bbb6b71649eca；所有分片/整包、本地逐轮CSV、源和质量匹配证据已核验，完整原件目录service-results/20261013_tianshu/decode_coherent_official4k/official4096。Git CSV为LF副本，原件字节保留。Math归档SHA55632c9f4b5ba32532507dbafdab57ecba6b30660971cc1d0be9c359470460eb已独立核验。

匹配启动脚本：experiments/tianshu_decode_tiles_coherent_20261013/serve_tianshu_minicpm.sh。其CLI和11项计算/路由开关与实际Math启动记录相同，另从实际进程核验compact varlen=1及PYTHONPATH。默认4K采用本配置，旧tag tianshu-best-buckets75-gather-20261013保留为回退。当前profile没有正式16K结果；以前2665.42属于另一套Prefill stage2/Decode32位地址配置，不能合成一套成绩。按用户要求后续只做常规4K测试。

结论：正式4K与独立质量均通过，刷新4K最好。硬件性能剖析未采集，TODO。后续独立探索更大Decode tile，保持本次已验证源码和配置。安装源、权重及官方benchmark不改，沐曦只读。
