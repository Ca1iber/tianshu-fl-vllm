# 天数BI-V150 32g：18点Prefill分段图跑通但无吞吐收益

新候选仅FL GraphWrapper输出采用独立验证的ATen非持有别名，实际图捕获18点，全部83点GEMM warmup保留。FULL_AND_PIECEWISE、mem0.70、4096预算、.cv/S8/KV16/nativePrefill/nativeTopP、C64/N64/4K输入1K输出、temperature1/top_p.95。注意旧tracer的configure83记录不代表83图；实际worker捕获18、policy与83次warmup分别核验。

复用先前同0.70 control，仅作为初步参考，不是新鲜同期A/B；control3804.46排除/4061.78/4054.13，稳态4057.954927。候选2913.33排除/4067.68/4019.58，稳态4043.631526，差异-0.352971%。候选首轮TTFT36.805秒，第2/3轮11.948/12.314秒。未达到进一步确认门槛，不采用、不做该配置新Math或正式benchmark。不能将小幅负值归因为已证实的硬件机制。

复用参考192+候选192个请求全量usage4096/1024、唯一种子、文本/TTFT/wall time、吞吐已重算。20冻结源、两组6安装源、实际.cv生成字节和producer/reducer匹配、实际capture18/warm83、Graph输出别名cpp/so身份、KV>=327680、全部12份metrics零抢占、GPU清理本地核验通过。两次steady候选差约1.2%，不声称比原配置稳定。

本轮工程进展：修复前0.70在2/83捕获OOM；非持有别名后37/83仍OOM；缩为18捕获后真实服务跑通。完整83预热从未缩减。资源可行性改善没有转为吞吐成绩；没有新正确率结论，不用于替换当前正式best4091.29/Math103of105。

完整原始5300081bytes/21parts，SHA256 4177a41a8cd12449af8b68707a754fe40d3c303fb7c5de5bd6dfae6497d48677。raw在service-results/20261014_tianshu/piecewise_sparse_candidate/20261014_piecewise_sparse_candidate_tianshu。模型profile未采集。下一独立方向为当前.cv主体K/V不对称cache和eviction策略，不重跑16K。
