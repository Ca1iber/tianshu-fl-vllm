# 天数 BI-V150 32g：真实IR RMS服务收益只有0.23%

保持当前TopP w16/stage1、Decode L0/KV32、native Prefill、B75G1/budget2048，只有候选IR优先级改为tianshu_ir_rms+native并使用已验证opaque实现。4K输入/1K输出、C64/N64，两组各3轮，预先排首轮。控制3529.690341→候选3537.853846 total tok/s，提升0.231281%。未达到足以继续全量评测的收益，不采用、不跑新Math或正式benchmark。

|轮次|控制tok/s|候选tok/s|控制TTFT ms|候选TTFT ms|计分|
|---:|---:|---:|---:|---:|---|
|1|1309.506670|1298.988016|18943.586568|18993.191761|否|
|2|3539.743255|3537.545244|13279.991122|13178.721958|是|
|3|3519.637426|3538.162448|13354.591184|13201.044220|是|

384/384请求成功，所有usage4096/1024/5120及逐轮配对提示种子核验。7个greedy探针一致；它不覆盖随机采样，不能替代新Math。真实150组normal/fused记录source790ca，候选确实命中；控制没有IR记录。两组TopP16及Decode32/source相同，六份安装源匹配；三轮服务不是此前未命中实验的控制复用。

两组都显式关闭CoreXOOT未导入的fuse_norm_quant pass，保留Inductor及FULL_DECODE_ONLY。除候选kernel-config IR优先级外，实际server argv完全相同。数值170项/epsilon1e-6通过且模型150组绑定已证实，但端到端收益小，不能将算子级优化或绑定成功写成模型大幅加速，也不能据此断言所有RMS优化都无收益。

完整原始包4091910bytes/16分片，SHA256 d7cf5fdcafb4c2588a6e98b71f7f3f58e7fe4d3bc989d70223d81c0d992df1c1；分片/整包、384响应、7份冻结源、真实IR/TopP/Decode记录、两组CLI和均值已本地核验。原件service-results/20261014_tianshu/ir_rms_service/20261014_ir_rms_service_tianshu。

结论：暂不采用此IR RMS实现，停止本方向全量评测，继续寻找其他框架/算子优化。当前正式最好3506.84/TTFT4494.12/Math103，tag tianshu-best-topp-w16-20261014保持。本候选没有新Math，不能复用旧103。性能剖析：TODO图内kernel覆盖；没有新正式16K。安装源、权重和原benchmark保持，沐曦只读。
