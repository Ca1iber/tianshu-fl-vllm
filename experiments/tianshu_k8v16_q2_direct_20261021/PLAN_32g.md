# Q1/Q2直接读取压缩KV的组件实验

只改变已合格pair-rows reader包装器的3D准入上限1→2，所有设备函数/codec保持原AST。当前正式模型拒绝投机配置，所以本实验不是声称线上已执行Q2，也不修改407/450-source模型。现有Q>1路径会经BoundedDensePrefill展开BF16；uniform Q2基准用真实同一gather arena和hybrid FlagGems BF16 prefill入口，包含展开成本。混合Q1/Q2只做正确性，不用扩大gather范围的简化baseline宣称服务收益。

显式按query token而非sequence数分配中间缓冲，最多128token；原64token缓冲不能直接复用。实际请求数和ragged CU不补假请求，query因果位置=KV长度-query长度+局部query序号。测试打乱页表、非连续Q/输出/物理页、两次改变Q/K/V/页表/有效长度，并做真实CUDA Graph重放。

GPU全输出与解码后的K及原V做FP32 attention对照，另CPU抽查原Q/K/V的3个序列×4head×每个query，保存真实timed输出及CPU抽查输入。比较T16/W1、T32/W2、T64/W4三种3D配置；uniform Q2记录真实gather+BF16 prefill对照。每组件9次交替×20重放取中位；任何数值/边界失败保留原始failed，不修改脚本重跑。

必须等当前Math8、C64补充质量、正式4K串行runner均终态，且ixsmi确认空闲才手动启动本组件。截止2026-10-21T15:58:00+00:00，最晚到时alarm退出。启动前冻结py/sh/CPU门控和本计划。此组件即使加速也不能单独采用；模型整合还需处理投机守卫、真实混合图、M128验证GEMM精度和新鲜Math>95%。
