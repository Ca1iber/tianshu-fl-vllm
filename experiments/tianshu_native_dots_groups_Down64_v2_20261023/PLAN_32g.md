# 原生W4分组v2：仅64/128，与BF16配对

首版group256实际运行时因不在允许集合(-1,32,64,128)被拒绝，未产生数值或计时数据；首版源码、失败与CPU输入全部保留。本版只移除256操作，复用已完成42层group64及128 bank与3层×3phase完整FP64参考，不重建大数组。所有参考、量化方法、数值阈值不变；未将未通过的group256冒充可运行。

M64 K6144 N2048，42层不同真实bank及3phase GU派生H、逐层roll。实际r3 BF16对照、nativeW4 G64和G128；全部42×3输出NaN-poison graph/eager逐字节一致、有限，3层×3phase完整FP64阈值abs<=.012abs(ref)+.006RMS且relativeL2<=.006。完整输入/bank/scale/zero字节/版本/guard、RNG/checkpoint/source保持。记录量化投影误差，质量未跑不得借用。

Graph/eager各7轮排首轮后6轮中位数，10次全部42层，循环旋转并交替反转候选顺序。Graph主要判据，eager分配差异明确。CPU独立复核后才判断是否集成模型。原始数组只容器，代码/报告/关键结果推天数GitHub。GPU前冻结，900秒上限，15:58UTC硬停止；不跑模型、正式benchmark、16K。
