# 天数 BI-V150 32g：Decode KV32两warp微测耗时下降约8%

在当前L0 KV32/w4/stage1对照上，只补测warp2/stage1、warp1/stage1和warp2/stage2。生产与reduce_segments的KV TILE始终同步32，网格/整数宽度/指针/模型权重不变。8个长场景和14个短边界，共66候选场景全部通过GPU FP32、选定CPU独立参考、native逐位比较、输出/工作区尾部与Graph捕获重放；没有编译错误。

|B|KV最大长度|warp4 ms|warp2 stage1 ms|耗时变化%|
|---:|---:|---:|---:|---:|
|1|4096|0.035218|0.038025|7.970502|
|8|4096|0.161471|0.149867|-7.186709|
|16|4096|0.348795|0.325132|-6.784387|
|32|4096|0.576242|0.525825|-8.749364|
|64|4096|1.069009|0.984209|-7.932550|
|64|5120|1.324800|1.222373|-7.731521|
|16|16384|1.367968|1.263733|-7.619663|
|2|32768|0.410317|0.343702|-16.235003|

9轮交替Graph样本均保存，每轮每图50次重放。B64/4K为1.069009→0.984209ms（-7.932550%），B64/5120为1.324800→1.222373ms（-7.731521%）。B1反而慢7.97%，因此服务候选仅batch8..64用warp2，其余保留warp4；warp1/stage1与warp2/stage2不选。32K和16K行只是单算子正确性边界，不是新的16K服务benchmark成绩。

此为局部算子收益，不等于模型吞吐已提升。实际服务三轮4K/1K C64 N64同期对照另跑，基础TopP16/nativePrefill/B75G1/budget2048保持。若端到端收益明确，再新独立Math105>95并正式4K，不复用旧103/105质量。

probe SHA fbf8c16888fc047c2bb67fda07f8da2b5dac79937baf0d52a4bbc5b72b84692e；native bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0，零初值生成源4a1f8bdfdb5b9399702941a0e499e62b52f104737aba6012a9056ef36f5c4d4e。实际compiled num_warps/num_stages/shared与生产合并tile记录保留；没有硬件profile计数或图内时间占比，TODO。

|全原始归档|bytes|SHA256|
|---|---:|---|
|long|2455182|d793e722d9168a7c48cce8d714017c052d014df106466d6f84a7b43bb61ca35b|
|short|2457535|73f0055d3d3435bf3c3597bec9aaef6667bb1318dcb522cdd5d66083269dde6c|

两份10分片及整包SHA、本地22场景/66候选、probe/router/helper与native源快照、原始timing和CPU检查全部核验。原件service-results/20261014_tianshu/decode_low_warps_{long,short}。当前正式最好3506.84/TTFT4494.12/Math103/tag tianshu-best-topp-w16-20261014保持；没有新Math或正式分数，沐曦只读。
