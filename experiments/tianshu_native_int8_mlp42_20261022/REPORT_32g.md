# 全42实际bank含Down holdout：仍有收益，待模型验证

固定上一轮NN/ROWS1/BLOCK1024/W8，仅M512/4096，没有再选配置。全42层原W8 GateUp与BF16 Down，GPU银行签名前后相同，原checkpoint SHA不变。当前GateUp与候选epilogue均严格匹配上一组件的实际计时binary；双方含相同A8及实际FlagGems BF16 Down。

|M|当前完整MLP每层ms|原生INT8完整MLP每层ms|逆耗时改善|每层绝对节省ms|
|---:|---:|---:|---:|---:|
|512|0.535758|0.504657|+6.16%|0.031101|
|4096|3.059375|2.591161|+18.07%|0.468214|

7轮交替顺序×3次整Graph，每图42个不同bank，计时除42。输入X在bank间共享合成fixture，不是42层完整Transformer串联，也不是4K服务吞吐。所有运行时量化、原生INT32中间写读、epilogue和Down均计时；CPU静态pack/诊断排除。

2个M×3变化输入×42bank=252次完整GPU GateUp输出/INT32/投影/Down逐位检查。每bank独立Down输出，整图前NaN填充，重放后全部42bank完整hash与eager相同；不是只看最后bank。独立CPU重建全部42个权重/scale/Down签名、28311552个完整A8元素、6144个selected INT64结果、1536个selected Down FP64结果及投影/SiLU选点通过。CPU整数/Down是选点，完整矩阵对照在GPU；24个selected bank artifact均hash校验。源/cache/权重与RNG/guards不变。

NN复用现有bank，不新增另一权重layout。这里组件bank/GPU scratch/独立输出约数GB，为验证设计；不能推断真实模型可按同样方式常驻。实际服务应把native workspace作为customop临时张量，并让原生Worker.determine_available_memory中的profile_run实际执行M4096新路径，在分配KV cache前观察到峰值。不能等cache配额后再新增不计费常驻缓冲。

下一步真实模型fake/FX与内存审核、短4K同期对照，再fresh官方C8及实际C64 Math，最后才正式N256；目前无新模型吞吐或质量成绩。M8/M64及其他M不改变。正式仍5695.95 total tok/s。

raw `/workspace/benchmark-results/20261022_native_int8_mlp42_GPU_tianshu`，代码/计划/报告/关键JSON/CSV在天数GitHub，完整数组/日志/cache仅容器。
