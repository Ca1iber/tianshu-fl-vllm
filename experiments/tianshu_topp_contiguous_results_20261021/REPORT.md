# 连续布局 top-p 对照完成：16 warp 过滤时间下降约28–35%，模型尚未验证

原生数值JIT及8192/4096 tile不变，仅warp2/4/8/16对照。此轮实际FP32 row stride130560，guard在完整首尾行，已由独立CPU检查metadata；与上轮stride130567压力场景分开。12场景、3次变输入、5模式180全输出通过重建SHA核验，合计846,028,800个FP32值；1272行CPU理想nucleus诊断保留。输入/全部guard/RNG保持。7轮轮换×20次Graph replay、420计时。每次完整恢复原始未过滤logits，各臂含相同copy，不含RNG/完整softmax/模型。

| C64场景 | 原生4 warp ms | 16 warp ms | 时间下降 |
|---|---:|---:|---:|
| normal1 | 6.588740 | 4.276324 | 35.10% |
| normal4 | 4.322644 | 3.108076 | 28.10% |
| peaked | 1.696111 | 1.131965 | 33.26% |

默认warp4和显式warp4 binary SHA相同；warp2/8/16对原生都有部分mask差异，逐位资格仍false。没有重新定义旧测试为通过。寄存器计数193→70、spill均0，只是单线程资源记录，不据此推断生产占用率。

独立布局比较完整读取36对原始BF16输入和p，180对输出SHA与全部mask完全相同。因此上轮[数值差异分析](../tianshu_topp_launch_results_20261021/verified/CUTOFF_SUMMARY.json)与[TV分析](../tianshu_topp_launch_results_20261021/verified/TRUNCATED_TV_SUMMARY.json)精确适用于这些相同输出，未重复GPU或假称泛化。p≈.95共1208有效行，warp16对原生32行不同、无rank inversion，最大TV .00186977、平均.0000140734。原生自身也有FP64 ideal差异，全部保留；不以此宣布所有新配置数学等价。

原始GPU目录 `/workspace/benchmark-results/20261021_topp_contiguous_GPU_tianshu`，raw SHA `b4f8818c269c15199d7b5b80c465321a311b389b547ba4f3985cd99f7001bbea`。10项冻结源全部复核；GPU/py/sh/CPU_GATE未更改。全部CPU理想逐行明细仅存容器 `/workspace/benchmark-results/20261021_topp_contiguous_CPU_tianshu/NATIVE_IDEAL_NUCLEUS_DIAGNOSTIC.json`；Git只关键gate/计时/对照证明和代码。GPU已空闲。

当前不采用，无新Math或整模型吞吐，推荐仍5586.28。下一步按用户允许的严格>95%质量标准新建明确非逐位模型候选：控制继承当前407源优化，唯一改变纯TopP线程数；实际模型布局/路由、预热原生显存计费与默认RNG必须证明，小4K A/B显著再新Math105与正式4K。详见 [实施约束](../tianshu_reservation_20261021/NEXT_TOPP_SERVICE_PLAN_20261021.md)。
