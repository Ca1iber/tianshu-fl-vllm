# Down64原生W4分组v2：group64数值通过但更慢

首版group256实际运行时拒绝已单独保留。v2冻结053384b，仅比较group64/128与实际r3 BF16，复用全部42层真实bank和原完整FP64参考，未重建大数组。

三种路径全部通过独立CPU复核，3538944格完整FP64参考；全部42层×3phase输出finite、poison后Graph/eager逐字节一致，完整输入/权重/scale/zero字节和版本/guards、RNG、source、checkpoint不变。

Graph 7轮排首轮后6轮中位数（每轮10次全部42层）：BF16 2.317032814ms，G128 1.969928741ms，G64 4.908340836ms。G64耗时约G128的2.49倍，热轮spread仅0.00580%，明显负结果；其最大投影relativeL2从G128的13.8703%降至12.2285%，静态误差改善不能当模型正确率。eager同样负向，G64 5.011208534ms，G128约2.041ms。

不接入模型、不跑Math/正式benchmark、不采用、不原样重跑。组件G128对BF16的约17.62%速度差不是新模型成绩。正式合格最好保持r3 5950.75 total /1190.15 output tok/s、TTFT3174.58ms。原始数据仅容器`/workspace/benchmark-results/20261023_native_dots_groups_Down64_v2_GPU_tianshu`；源码、独立CPU门禁、关键JSON/CSV入天数GitHub。
