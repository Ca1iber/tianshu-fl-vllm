# 后续候选草稿：在Down64 W4上另加LM64 dotsaddz W4

当前正式Down64单项评测继续原样运行，此目录仅CPU准备、没有GPU执行。LM64组件曾完整数值通过，相对实际已启用W8A8核心省0.0915ms（17.61%），未证明整机收益或质量。新候选只在原dense opaque-op的真实LM bank且M64路由W4，M8/其他M与GU全部委托真实原backend，不改schema/fake或apply。保留原checkpoint、BF16及W8；新W4 buffer在实际load profiler内注册，warm与profile scratch probe均在真实native worker memory profiler中。

必须先完成新独立审计适配（显式以新LM64 W4真实执行代替旧LM64 W8计数要求，不伪造原记录），CPU门禁与冻结，再新的完整C8/C64质量。当前没有CPU_GATE/EXECUTION/quality controller，不可运行。只有质量>95%、短4K通过才考虑正式；当前r3与单Down正式冻结文件都不修改。
