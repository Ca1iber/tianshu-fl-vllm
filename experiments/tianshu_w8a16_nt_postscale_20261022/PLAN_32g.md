# W8A16 NT 权重布局、dot后统一缩放

前一 W8A16 NN 后缩放版本的 Down 初步计时仍慢于 BF16，先不归因硬件。本独立候选将 INT8 权重由 [K,N] 改为 [N,K]，沿用每输出通道 scale、activation BF16、INT8码精确转BF16后dot、最后FP32 scale。与NN后缩放为相同数学定义，实际布局与三组预定tile整体对照；不把综合差异归因单一因素。

真实Down/LM head M8/M64和M17/N131/K259 strided/tail case。tile为BM16/BN64/BK128/w2、BM32/BN128/BK128/w2、BM64(若M64否则16)/BN128/BK128/w4，均1stage。三输入相位、全量格式decode、完整GPU参考、独立CPU FP64/全部输出cast/原W8A8整数结果与7×20 Graph计时沿用。新数值路径不能借用旧Math；阳性组件才考虑引擎实际内存和fresh Math。

前任务结束、审核完成且卡空闲后运行，暂未加入现有队列。每天北京时间06:00–次日00:00，本日23:58截止，单任务最多30分钟，原source/失败不改，原始weight只读。raw只在容器，代码/报告/关键结果推天数GitHub。
