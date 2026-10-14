# 天数32g：准备Decode K/V读取缓存策略短诊断

检验当前长KV读取的cache_modifier参数，不预先假定天数L1/L2结构或性能收益。候选只对3D Decode的两次K/V tl.load添加显式策略，比较原L0、带新constexpr的默认空策略、.cg、.cv。2D/Prefill继续空策略，其余形状、数学、类型、grid、KV tile、warp、stage及B64S8完全保持。

CPU来源AST复核：移除一个新增constexpr和两处tl.load keyword后与原L0整文件AST完全一致，原版权头保留。生成源SHA8deb8967...；764组实际Router参数传播/网格/reducer通过。GPU支持性及效果仍未测，不能由通用Triton语法推断天数后端有效支持。

GPU计划17场景、默认copy/.cg/.cv三候选；改变捕获地址中的query与K/V值后，重新计算独立GPU FP32/CPU参考，检查缓存可见性和尾部，不只用静态输入重放。保留编译/数值失败。B64/4K各选项保存编译产物及实际寄存器信息，但二进制hash不同不能单独证明cache hint在硬件生效，不从事件时间推断缓存命中率。

目前未排队。只有预算实验无明确候选且寄存器诊断完成后才考虑该短探针。当前正式最好3826.24/Math102保持，无新吞吐或质量结果。

2026-10-14 13:41+08更新：4096预算独立确认收益3.34%，当前新Math/正式4K仍优先。post-quality串行控制器PID195970已实际存活等待quality chain193778退出并完成正式归档；之后先寄存器，再执行本缓存诊断，最后M4096 GEMM。不并行占GPU，活动26源冻结。新增下载复核器检查7份来源、17场景、变query/KV的GPU/CPU数值、原始设备event及编译产物SHA；目前仍无本候选GPU结果。
