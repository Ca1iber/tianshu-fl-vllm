# O64 非对称W4：真实模型启动通过，质量评测中

2026-10-24，BI-V15032GiB。基于已采用r4，仅在FULL uniform64 decode将42层O投影切换为原生G128 affine W4A16。QKV64仍BF16，其余prefill/Down/LM/GateUp/KV/采样路径保持原源。959份模型来源，原r4全部944份SHA保持，原13项serve参数保持；实际65项环境包含新O64模式和记录目录。

所有42层原BF16及新W4/scale/zero完整字节签名与组件证据一致。42个新O64捕获、42个QKV64 BF16捕获、原GU融合及Down64 W4捕获通过独立检查。42层新增buffer共93,585,408字节全部纳入native DeviceMemoryProfiler；模型统计7,905,652,224字节，相对r4正好多93,585,408字节；全部模型唯一参数/buffer存储7,836,359,328字节。42层native scratch在原生峰值分析窗口中探测，KV物理容量自然变为583,664tokens，没有手动放大KV预算。旧BF16/W8副本保留，不声称净显存节省。

启动检查完成后已用全新API开始C64 Math105，FutureKV为control以确保实际并发覆盖。C64结果尚未产生，没有新模型正确率或吞吐成绩，当前最好仍为r4的6050.94 total tok/s。后续C64严格>95%再跑原C8，并做4K短测；不跑16K。全原件留容器，代码/报告/关键结果进入GitHub。
