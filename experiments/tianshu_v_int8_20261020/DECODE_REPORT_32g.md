# BI-V150 32G：线性 INT8 V 的 paged decode 组件为负优化

结论：8 个场景、每个场景 4 种缓存、2 个 Graph 数据阶段的数值与存储检查全部通过，当前线性 INT8 V 的单层 decode 在所有场景更慢，不进入服务/Math。正式最佳仍为 K8V16/default4096 的 5101.45 total tok/s；本报告没有新服务分数。

## 对照与关键结果

同进程控制组使用实际已验收 K8V16 reader（SHA256 dc69f525d8c7686a8977f50e901af39820ce92486894890631c53fc01dc46700）和原始 Producer/Reducer。新启动器的 AST 检查确认仅允许 UINT8 V 类型，其余启动与 reducer 不变。K 编码相同，物理页由 12288 降至 8192 字节，候选 V 缩放为 1/32、1/16、1/8。

以下为 V=1/32，单位 ms，计时包括一次 KV writer 和一层 paged decode 的热 Graph。每项 9 次交替顺序，每次 20 重放，原始 event 和全部三档数值见 decode_timings_32g.csv / DECODE_GATE_32g.json。Graph 第二阶段把长度减 7，因此 4K/5K 行的实际计时长度分别为 4089/5113；两组输入完全相同。

| Batch | 上下文上限 | K8V16 | 线性 V INT8 | 耗时增加 |
|---:|---:|---:|---:|---:|
| 1 | 4096 | 0.035970 | 0.041610 | 15.68% |
| 1 | 5120 | 0.040293 | 0.047307 | 17.41% |
| 8 | 4096 | 0.088848 | 0.115365 | 29.84% |
| 8 | 5120 | 0.106303 | 0.139070 | 30.82% |
| 64 | 4096 | 0.589235 | 0.777874 | 32.01% |
| 64 | 5120 | 0.729068 | 0.963127 | 32.10% |

另外两个 ragged/strided 场景也更慢约 26.01% / 30.01%。三档缩放结果相近。这里的“耗时增加32%”相当于该组件处理速度下降约24%，不能混用两种百分比，更不能直接换算为完整服务吞吐。

## 正确性范围

64 项全输出 GPU FP32 grouped-GQA 参考检查、640 项所选序列/heads 的 CPU 参考检查通过。32 个 Graph 各重放两个数据阶段；同一 Graph 改变 Q/K/V、block table、lengths 与 slots 后，完整 cache storage、输出和 segment scratch guard 仍正确，RNG 状态不变。包含跨页、输入/table stride、负 slot 和非均匀长度。

V 随机 fixture 的越界和误差统计均保留在 JSON。这些是随机输入，不能代表模型真实 V 分布或 Math 正确率。本轮没有连接引擎 allocator、prefill、模型或服务。

## 代码生成证据与下一步

实际编译记录中 B1/B8 的寄存器从 143 增至 184，B64 从 142 增至 183；均无 spills。tile16、warp1、stage1、segments、grid 相同。抽取的实际 B1 LLVM 出现 sitofp i8，汇编包含逐元素 ml_cvt_i16_i8、ml_cvt_i32_i16 扩展链；IR、汇编和摘要已保存容器 codegen/，关键资源与转换计数见 CODEGEN_AUDIT_32g.json。目标为 corex arch71，warp_size64。

这支持“额外转换及寄存器压力可能吃掉存储收益”的解释，但未测 achieved occupancy，不能把性能下降完全归因于占用率。下一步只尝试一种避开有符号整型转换链的等价格式解码，先做全部码点证明与小组件对照；没有真实收益则终止该 V 压缩路线，不为负优化投入正式 benchmark。

原始数据与冻结源码：/workspace/benchmark-results/20261020_v_int8_decode_tianshu；执行区间 2026-10-20T01:35:44.587829+00:00 至 01:36:12.240868+00:00，不含此前插件初始化。本报告不宣称冷编译或服务端到端耗时。GitHub 只保存关键 JSON、CSV、报告和实验源码，完整编译输出与原始小输入留容器。
