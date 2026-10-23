# LM全小批次W4新模型质量独立CPU复核

在新模型C8 GPU质量评测仍运行、结果尚未知时冻结此验证器。完整105×2逐题review/prediction/report重算，不过滤caps/errors；同944模型源、13argv/63环境、同controller版本，C64仅FutureKV control；全部当前源与实际GPU源/控制器快照逐SHA复核，源只属于本次新模型不能借用旧934质量。真实LM8/64 W4调用、旧BF16/W8未用、新pad kernel和bank签名/native真实峰值、Down42 capture及C64 Graph联系、QKV/O fallback等原检查保留。

CPU独立重算每题得分、105分母、停止原因、原始文件SHA和checkpoint；再重放已冻结的完整运行审计并与GPU评测的accuracy_audit.profile逐字段相同。不把复用的运行审计说成另一份数学实现。两个batch>95%才model_quality_qualified，单batch报告仅是阶段结果；component数值不当模型正确率。原始不改、不重跑GPU。

支持--batches 8或64输出阶段门禁和CSV，默认两者生成最终门禁；任何结果文件已有则拒绝覆盖。质量不达标同样保存真实不合格，不能借用旧版成绩。JSON/CSV/代码/报告推天数GitHub，完整raw/数组只容器。本验证器不跑GPU、Math或吞吐，不修改模型。正式最好仍r3，只有新完整质量+短测+正式门槛都过才能替换。
