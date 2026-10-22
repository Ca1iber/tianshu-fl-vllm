# 原生INT8预填充：全42实际bank、含Down的固定候选holdout

前组件第0层NN/ROWS1的完整GateUp M512+12.43%、M4096+31.12%，已通过独立CPU及严格正式控制binary。固定这两个M、NN物理layout、epilogue ROWS1/BLOCK1024/W8，不在本轮再选配置。复用现用W8编码及scale，所有42层GateUp和BF16 Down均读取原checkpoint、全签名复核，不新增NT weight副本。

控制：旧A8+逐M正式two_dot SiLU融合+实际BF16 Down mm_out。候选：同A8+torch._int_mm(out)+同精度/SiLU epilogue+同BF16 Down。逐层变化A的3相位，对全部42层核验完整GateUp/INT32/BF16投影/Down位一致、guards/输入/RNG。每个bank有独立Down输出，整42bank Graph输出先填NaN后重放，再对每个bank与eager完整hash核验，不能只看最后一层。7轮交替×3重放，每图42 bank，计时除42，包含所有运行时读写及Down。

输入X在42bank之间共享合成fixture，不是完整Transformer依次前向，组件cold-bank holdout不称为模型吞吐。独立CPU重建全部bank量化/Down hash、全A8、在层0/7/20/41选点INT64/投影/SiLU及Down FP64。原权重/源/cache前后hash不变；full INT32/输出比对在GPU，CPU整数dot只selected，不能混淆覆盖范围。

最大INT32 workspace201326592字节。组件自己的共享scratch/42独立输出用于完整性验证，不假设模型能以同方式常驻；真实服务必须让native profile_run看到workspace峰值，再分配KV cache。若本轮阳性再实施fake/FX/memory覆盖、短4K同期对照、fresh C8/C64 Math及正式benchmark；其他M和decode不改。GPU单任务最多30分钟，今日23:58停止，每日06:00–24:00授权，代码/报告/关键结果天数GitHub，raw仅容器。冻结后修复新版本，用户脏文件与沐曦仓库保持。
