# M4真实BF16 LM apply Graph差异诊断

原小批次首版M4 phase0在Graph/eager一致性或finite组合assert失败，未保存失败数组，不推断根因。仅诊断当前r3 BF16 EmbeddingMethod.apply，完全复用LM权重与3phase输入/原始完整FP64参考，roll1、M4；不启用任何新模型精度，不计时，不做benchmark。

每phase保存3次独立eager输出（先clone再保存）、3次未poison Graph replay输出、3次全NaN poison后Graph replay输出，全部先存数组和SHA，不因差异跳过证据。独立CPU分别统计finite、NaN/Inf、同eager逐字差、float数值差（排除signed-zero误判）、最大误差、原始FP64沿用阈值检查。零输入行期望0，逐行另列。完整输入/weight字节版本guard/RNG/checkpoint/source仍须不变。输出差异本身是诊断结果，不能当作已通过的模型新优化或放宽原失败门槛。

首GPU前冻结，300秒超时，15:58UTC硬停止，原数据只容器，代码/报告/关键结果入GitHub。后续根据实际差异制定新实验，不原样重跑挑好结果。
