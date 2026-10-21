# 少量多查询复用原生 SplitKV：独立组件计划

2026-10-21。当前推荐版5586.28不变。旧DSpark2步概率/greedy均实测负收益；当时未使用当前K8V16与INT8路径，不能直接叠加。当前byte reader wrapper max_seqlen_q>1强制2D，而legacy Producer给所有多查询BM256/Q32/T128/16warp，padding与长序列串行循环可能过重。原数值JIT/codec/掩码/reducer已经支持多查询。只研究wrapper开启有足够scratch、Q<=8的分段路径，加六种预声明query tile，不改任何数值JIT AST。

对照：原S16 codec、12KiB K8/V16真实物理页的2D byte fallback BM256/Q32/T128/w16。候选：原JIT启用3D，BM32/Q4或BM64/Q8；T32/64，w2/4/8，S采用当前按N的分段数（大N短K原kernel实际固定4段）。所有关键地址仍int64，K量化及BF16 V不变。当前prefill arena能容纳时会走gather+dense，不在本轮基线内；组件结果不得代表完整当前路由或投机服务收益。

先CPU验证原生query block二分映射、token/head唯一覆盖、segment区间以及源码可逆与数值JIT AST完全保持。GPU检查全输出对解码后的FP32注意力参考、选中CPU参考、两次变Q/K/V/length/page table后同Graph重放、strided与guard、完整cache无写入、RNG保持。所有模式七轮旋转次序、每轮20次Graph热重放，仅单层attention读取；不计静态fixture创建，不含KV writer、模型GEMM、采样、草稿、拒绝采样或服务端到端成本。短长度/参差查询检查掩码与未活跃segment，不做16K。

scratch按真实总query token数分配，不能把原64-token scratch直接用于64*8查询。Q8/N64/S4额外约17MiB/layer invocation共享；小N/S64同样按量核验。未接模型allocator、未改变精度或推荐版本；任何后续模型接入均需原生显存计费、target/draft分别适用、真实Graph、正确标准拒绝及全新Math门禁。硬截止2026-10-21T15:58Z。

启动前页表安全核查：原reader对page-table索引无mask，T128末tile最多再读7个无效页ID。fixture每行至少分配logical_width+9的真实存储，logical_table保留原width，尾部为负poison；不越过已分配页表，KV数据load仍由原tile_mask保护。该条件也必须在未来模型接入审计中成立；不能把logical shape等同实际分配边界。
