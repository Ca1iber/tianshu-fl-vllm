# 原生 Qrita top-p warp 配置：局部提速，未达到逐位原生等价门槛

只改变 num_warps，原始数值JIT、BLOCK8192/TRUNC4096、TopP/duplicate/非有限值逻辑不变。B8/B64×6分布×3次变输入×5模式，180个完整FP32输出通过保存mask与BF16输入重建，全部846,028,800个值的SHA一致；CPU另核对1272行理想FP64 nucleus。全部输入/RNG/guard保持，7轮轮换×20 replay共420计时。每次replay先完整恢复未过滤logits，避免反复过滤的假速度。仅过滤与恢复，不含RNG、完整softmax或模型，不是吞吐/Math成绩。

**布局限制：本实验所有工作logits的row stride为130567（词表130560+7 padding），属于带行间隔压力场景；不能当生产连续FP32 stride130560的实测。下一份新组件将补连续布局，旧源/结果/gate保留不改。**

| 场景 | 原生4 warp ms | 16 warp ms | 过滤时间减少 |
|---|---:|---:|---:|
| B8 normal4 | 1.070090 | 0.764703 | 28.54% |
| B64 normal1 | 6.549094 | 4.333557 | 33.83% |
| B64 normal4 | 4.411391 | 3.161011 | 28.34% |
| B64 peaked | 1.767107 | 1.175793 | 33.46% |

原生隐式warp4已由真实compiled metadata确认，显式warp4与原生binary SHA相同、全部输出逐位一致。warp2/8/16均有部分输出不一致，原逐位等价资格均为false，不改判通过。16 warp单线程register计数70，默认193，二者spill均0；这不等于整个CTA寄存器或生产占用率已经证明改善。

独立CPU差异分析在实际FP32 p≈.95的1208个有效行上发现：warp16有32行mask不同、共592个元素变化，无rank inversion；与原生截断分布最大TV 0.00186977、平均TV 0.0000140734。warp8有27行不同，最大TV同值。完整参考见 verified/CUTOFF_SUMMARY.json 和 verified/TRUNCATED_TV_SUMMARY.json；原始逐位失败仍保留。原生本身也有与理想FP64最小nucleus不一致的情况：p≈.95有330行mask差异、5行保留mass略低于p，最坏短缺0.000760746；其他p/异常分布另列，不能假称原生与理想全部相等，也不能据此自动宣布新候选正确。

原始GPU目录 `/workspace/benchmark-results/20261021_topp_launch_GPU_tianshu`，raw SHA `da35fa2935cc69f394d61117c28dd9da66210cc073df681289d8adc0ca99ac68`；10项冻结源含CPU gate，全部复核无修改。CPU全部行差异明细在 `/workspace/benchmark-results/20261021_topp_launch_cutoff_CPU_tianshu/analysis.json`，Git只保留摘要/代码/关键计时。当前不采用，不修改推荐版。

考虑用户允许以严格>95%准确率判断数值优化，不能只因FP32归约顺序改变就永久丢弃候选。后续另建明确非逐位候选：先测真实连续布局及真实服务logits，再做独立模型4K小A/B，收益明确后新Math105（严格>95%）与正式4K。旧组件的逐位门槛未过这一事实不修改；新模型资格是后续新证据，不能复用为旧测试“通过”。
