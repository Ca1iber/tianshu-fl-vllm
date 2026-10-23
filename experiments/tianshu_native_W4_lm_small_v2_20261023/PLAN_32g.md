# LM小批次v2：完整保存所有形状数值，单列不合格BF16对照

首版M4 BF16 graph/eager断言失败，独立诊断27输出全finite但FP64不合格且重复eager本身不稳定，旧失败不覆盖、不放宽原数值阈值、不冒充已通过。新版本完成M1/2/4/8/16/32/48/63全部组件数据采集；仅数值不合格继续记录，设备错误仍立即退出，不接模型。

复用原LM真实BF16/WD/S4/ZD和3phase完整FP64参考，row-roll1避免M1全零。当前真实控制为r3环境enable-all下EmbeddingMethod.apply，warm trace核对vendor linear。所有case图及输出保留至进程结束。每phase每变体先保存eager0，另外2次eager重算逐字节比对，若不同另存实际数组；再NaN-poison重放图，先保存实际数组再记录差异，禁止先assert丢失失败输出。输入/weights/scales/zero全字节/版本/guard/RNG/checkpoint/source仍必须保持。

全部eager及graph输出独立FP64检查，原阈值不变：relativeL2<=.006且逐格abs<=.012abs(ref)+.006RMS。所有不同重复eager输出均独立检查。CPU结果分别记录每个M/变体数值、重复确定性与graph一致性，不合格BF16点禁止作为合格性能比较；候选W4点必须数值、eager重复和graph全部过。不会把采集complete当全部数值通过。

每M eager/graph各7轮排首轮、后6中位数、20次调用，交替顺序。eager为LM当前主模型图外的主要组件指标；仅组件不等于模型收益，尚无新Math质量，不运行正式benchmark/16K。源码/计划/门禁首GPU前冻结；900秒上限，15:58UTC截止。代码/报告/关键结果推天数GitHub，原始仅容器，不原样重跑挑结果。
