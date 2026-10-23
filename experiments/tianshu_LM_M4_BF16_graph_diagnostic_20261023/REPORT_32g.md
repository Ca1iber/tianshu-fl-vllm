# M4 BF16真实LM apply：差异不只Graph，独立参考未过

冻结a7002b0，完全相同M4真实weight、3phase X，各3次eager/普通Graph/NaN-poison Graph，共27份完整输出先存数组，随后CPU独立审核。所有输出finite，无NaN/Inf；存在数千个BF16字差且全部对应实际数值差，不是signed-zero。重复eager自身也有差异，因此不能简单归为Graph生命周期或poison问题。

27份输出均未通过原始冻结FP64阈值（relativeL2<=.006且逐格abs<=.012abs(ref)+.006RMS），未放宽门槛。输入/原weight全字节、版本和guards、RNG/checkpoint/冻结source都保持。DIAGNOSTIC_GATE.complete表示诊断采集审核完成，并非all_FP64_pass；本次后者为false。

目前仅在此组件环境复现，不能直接推断已采用r3正式模型的LM执行也有该问题，更不能拿不合格对照宣称新模型提速。未计时、未改模型、未做新Math或正式benchmark。后续小批次v2保留每点eager/Graph全部关键输出及重复eager差异，不因一个BF16点失败漏掉其余形状；独立参考阈值不变，对照不合格的点不能作合格性能对比。

原始数组和硬件/源码检查仅容器`/workspace/benchmark-results/20261023_LM_M4_BF16_graph_diagnostic_GPU_tianshu`。代码、关键诊断JSON及本报告推天数GitHub。
