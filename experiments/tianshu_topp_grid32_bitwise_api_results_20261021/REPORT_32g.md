# TopP只加倍行间grid：全输出逐位相同，未获得明确组件收益

当前407源推荐版为父项，不沿用早期旧模型栈。保持GEO32K/16K、w16/s1每行计算，只将B8..64纯TopP的grid=min(B,16)改为min(B,32)，支持范围与原16K/grid2实验相同，fallback完整委托真实父Router。独立64行owned scratch防止跨批次图指针失效。

25分布×6p×2输入阶段，300组/600份完整FP32输出全部与同次原版逐位相同；33真实Sampler的mask、token、global/每请求RNG逐位相同。8跨批次旧图重放、8fallback、B8/B64 strided实际输入/输出均通过。CPU重建2879631360个FP32值并核验完整输出SHA，独立22056行FP64检查；candidate_changed_rows=0。原版自身严格阈值false共9行，本候选完全相同，因此两种实现合计保留18条false记录，不将它们视为新数值差异或删除。

500个交替热图事件包含全输入恢复；p.95常见BF16输入B64的phase0：normal 2.582698→2.595371ms（耗时+0.49%），peaked 2.049784→2.045867ms（-0.19%），strided 2.611376→2.620245ms（+0.34%）。所有分布、两阶段原始中位数完整保留在KEY_TIMINGS_32g.csv与GPU_GATE，不只保留较好点。

仅加倍grid未复现前轮16K/grid2的明显组件降幅，不进入模型服务/Math/正式4K。不能由本次结果断定所有并行配置均无收益。与前轮Math99/105质量拒绝是两项独立结论：本轮全输出与原版相同，但没有值得继续的性能证据。当前推荐仍正式5586.28 total tok/s、TTFT3780.63ms、Math102/105。

代码与完整CPU verifier在tianshu_topp_grid32_bitwise_api_20261021；GPU/CPU原始文件仅容器benchmark-results下同前缀GPU_tianshu、CPU_verified_tianshu。Git只存代码、报告、关键JSON/CSV，没有完整大tensor或模型权重。
