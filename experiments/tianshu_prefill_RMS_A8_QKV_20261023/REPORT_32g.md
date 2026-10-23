# prefill RMS+A8与原native QKV完整链：首次组件验证通过

| 真实QKV bank | 有残差 | M4096 control graph16 ms | candidate graph16 ms | 完整链延迟降幅 | eager降幅 |
|---|---|---:|---:|---:|---:|
| layer0 | 否 | 0.418645 | 0.392250 | 6.3049% | 6.5049% |
| layer20 | 是 | 0.471862 | 0.449186 | 4.8055% | 4.9529% |
| layer41 | 是 | 0.471836 | 0.447618 | 5.1327% | 5.1993% |

本轮首次GPU前a60a7bd冻结，固定18数值用例、180计时行和24个计时Graph，未同版重跑取幸运值。两臂完整链都包含norm、A8、原Torch CoreX _int_mm与原_dequant，只把候选norm+A8融合；原真实BF16权重、NN INT8 pack及scale签名、native MM与dequant顺序保留。eager与graph4/16均五轮排首轮、轮内交替臂顺序，warm10/replay20。M4096三bank的eager和graph16都超过预定3%改善，chain4/16最大差距0.24985%，通过10%稳定条件；M512只诊断，完整结果见KEY_SUMMARY。

独立CPU全量复核18用例、84,934,656个完整K FP64 norm单元，两臂均在预声明FP32/BF16界内，全部A8与scale对各自实际BF16 norm独立重算逐位相同。4608个完整K2048 INT64 projection参考单元的INT32和最终BF16结果精确匹配；明确这是投影抽样参考，不冒称整个投影矩阵独立重算。36个真正数值Graph和24个计时Graph返回对象完整毒化/重放，已保存全部输出字节核对，INT32全矩阵在GPU全字节核对并存完整hash及原始抽样数据。相同Graph/eager文件只在本目录内部共享字节，独立原始路径及实际Graph验证记录保留。

输入/原BF16权重/pack/RNG与所有可控制输出保护区通过；norm的M512/M4096×有/无残差实际CUBIN分别匹配原False/True二进制，两个QKV dequant CUBIN亦与原服务选择一致，全部无spill。全部代码/输入/首次GPU前快照及原checkpoint SHA复核，GPU前后空闲、原模型与结构未改。对照是独立编译本机IR-native函数，并未证明其CUBIN与整模型生成代码完全一致。

INDEPENDENT_CPU_GATE.complete及component_lead均true。这是组件完整链约4.8%—6.3%的GPU图延迟收益，不是模型吞吐成绩；当前没有prefill模型接入、新Math质量或新正式分数，正式最佳仍r3 5950.75 total tok/s。后续以合格r3为基础，仅新增4096静态编译点及QKV前norm+A8融合，明确真实42层/实际旧量化器计数变化和Profiler峰值，重新跑C8/C64各105题严格>95%，再同期短4K。

代码、报告、关键CSV及独立门槛放天数GitHub；完整输入/输出、INT32抽样、日志、cache及快照仅容器/workspace/benchmark-results/20261023_prefill_RMS_A8_QKV_GPU_tianshu。本地没有数组或模型副本。
