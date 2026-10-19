# BI-V150 32G：显式预算4096的新鲜Math通过

全新Math-500 Level3：102/105=97.142857%，105调用全部成功，0错误/缺失，严格>95%。独立保存并本地复核全部105 prediction、105 review、summary与准确率重算，207来源及实际41环境/argv/K8V16页/预算4096/warm83匹配。

103自然stop，2条max_tokens：索引83仍判正确，101判错。错误索引18、87、101。保留32768输出上限和所有原始文本，不删截断题、不重试替换分数。参数batch8、temperature1、top_p.95、timeout3600均按官方。

这是显式追加--max-num-batched-tokens4096的质量，不标作原serve argv，也不借给新增框架默认选择代码。对应3+3服务201来源/41环境的4896.5347→5048.0563（+3.09%）已独立通过，但还没有该配置正式N256成绩；当前原命令最好4997.05及Math100/105保持。

下一步已将选择落到EngineArgs原生默认边界，以原命令进行独立服务/新Math/正式验证。保留当前已通过CLI版本原件，不提前采用新增实现。

原始根目录 /workspace/benchmark-results/20261019_k8v16_budget4096_math_tianshu；档案6544059bytes/25parts，SHA256 3716a8fdff34bbdcca1b47f95641ce705f24732c45cb441f669325fb512d1cc2。完整及所有分片SHA通过，verify_validation_downloaded.py独立复核通过。没有运行16K，未采集硬件profile。
