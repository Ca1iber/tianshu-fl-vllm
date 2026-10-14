# 天数BI-V150 32g：最大序列数256与64服务对照

双方新服务3+3，4K输入/1K输出、C64、N64、temperature1/top_p.95、4096预算/实际83点/.cv/S8/KV16/SQL/原生TopP，候选只增加--max-num-seqs64。预先排除首轮，完整384请求。

|组别|首轮排除|第2轮|第3轮|后两轮均值 tok/s|
|---|---:|---:|---:|---:|
|control|3797.10|4053.30|4047.39|4050.35|
|candidate|3802.69|4093.02|4062.12|4077.57|

提升 **0.6721%**，低于预先设定2%确认门槛，暂不采用、不运行5+5或新的Math/正式benchmark。不能把本N64小测直接替代N256正式分数。当前已采用.cv最佳仍4091.29 total tok/s。

实际WorkerFL构造记录control max_num_seqs256、candidate64，二者budget4096、async=True、max_model_len131072。构造时两者均83个capture size、最大4096；因此本次没有缩小83点预热范围。服务日志control KV cache521024 tokens、candidate531296 tokens；仅是实际容量记录，不据此推断吞吐或按内存字节精确分解收益。

全部384原始请求的唯一seed/usage4096+1024/正文非空/TTFT及duration重算、16冻结源、6安装源、83实际预热、SQL/S8/KV16/reducer、实际256/64worker记录、两组命令仅max-num-seqs64不同均已本地复核。每组实际64条.cv生成路由的8deb字节和原L04a1f来源链也已重新渲染逐字验证。

完整归档5055769bytes/20parts SHA256 `c31ac98599b43929cf0537f970c3072369e560175740162fc8fff52e349b38c5`。原件service-results/20261014_tianshu/max_seqs64_service/20261014_max_seqs64_service_tianshu。复核入口verify_downloaded.py，以及.cv audit_generated_kernels.py分别对control/candidate。

保留已准备的严格门槛5+5入口，但本轮未达门槛，不启动。后续独立CTA排列微测已串行接续；没有与本服务并行争用GPU，没有修改安装源或沐曦仓库。
