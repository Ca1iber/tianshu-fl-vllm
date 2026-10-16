# 天数TopP自适应候选截断

假设：较尖峰的logits分布可以用更高的初始outlier cutoff，减少候选缓冲区及随后搜索的数据量。以当前严格32K几何搜索为控制，只在纯TopP分支将sigma改为max(原sigma, std_logit-bias)，bias分别3.0/2.5/2.0。候选概率质量不足时仍走原完整范围回退，最终精确搜索、重复值处理、32次上限、p与CPU1e-6门槛保持；TopK保留原kernel。

十场景已完成，三个候选的原始/变化输入strictCPU1e-6与Graph全部通过；3520条行语义、350计时及11源/4生成源已本机复核。bias2.5在B64尖峰1.687744→1.252588ms、BF16尖峰1.896377→1.672245ms，普通分布基本持平，见[报告](results/20261016/micro_32g/REPORT_32g.md)。正态近似仅选择初始cutoff，不作为概率答案或正确性保证。

已准备原始serve argv的独立4K/1K/C64/N64 3+3对照，两组都保留当前GEMM+geo最佳的38项profile，只增加一个自适应cutoff开关。实际解释器startup的CPU检查确认新TopP loader唯一、无Torch/GPU导入。服务收益尚未验证，不将局部时间变化当作正式成绩。
