# 原生 W4A16 Down64：组件快6.01%，尚不进入模型

结论：同一r3钩子环境和真实42层Down权重，CUDA Graph轮换42个不同bank时，BF16 2.320458 ms → RTN-W4/AWQ 2.188805 ms，速度+6.0149%，绝对节省0.131654 ms/解码步。普通eager为3.773118 → 2.283287 ms（+65.25%），其中包含Python/调度差异；服务主要使用图执行，因此主要依据图结果。两种模式均7轮交替顺序，每轮10次42层，第1轮排除，其余6轮中位数。没有运行整机benchmark，不将组件百分比当作整机收益。

2359296格完整FP64独立检查全部通过：0/20/41层、3个输入phase、原始BF16及量化BF16反量化参考。候选相对自己的完整FP64参考L2约0.166%，每格均在事先冻结的误差界内。全部42层、三个phase完整输出有限，eager与图poison重放逐字节一致；输入、原始权重、packed权重、scale、zero的完整字节和版本、边界guard、RNG及checkpoint/冻结源码均保持不变。

量化对原始BF16投影的最大相对L2误差13.8703%；这和kernel计算错误不同，但仍会影响模型精度，需要新鲜C8/C64 Math500。输入是旧真实GateUp权重产生的完整fixture，并非实际在线激活trace。量化是group128对称RTN，采用已验证的AWQ物理布局，未做激活感知校准。仍未改动原始checkpoint或服务。

执行前冻结commit：5d8015427fea8d01e801d43c92f81a48331d6551。903个父版本模型源及其控制器完整校验，实际baseline为 `_iluvatar.ops.mm.mm`，有效_launch_mm来自原r3 prefill钩子（其M2048以外分支回到原实现）。实际候选为 aten._weight_int4pack_mm_awq。没有用未知torch.mm调度代替对照。

原始数组、全部126对输出及42层packed权重仅保存在容器：`/workspace/benchmark-results/20261023_native_W4_down42_{CPU,GPU}_tianshu`。GitHub保留源码、CPU/独立结果JSON、关键计时CSV与SHA，不传大数组。结果JSON SHA为7950dc22402811c029948eae0431c36e12dd8d7a6bffb533ad79b649e81ee31e。所有14个计时记录可在原始GPU result.json复核。

不采用到正式模型：当前图收益绝对值较小，先探索厂商推荐dotsaddz路径/其他真实投影，若有足够收益再付出模型准确率与服务集成成本。正式最佳仍为r3 4K 5950.75 total /1190.15 output tok/s。本版本没有更改其成绩、质量结论、模型源或缓存。
