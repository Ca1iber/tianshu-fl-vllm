# 天数 BI-V150 32g：五项小收益的组合候选

当前正式最好4091.29 total tok/s、Math103/105。单项服务短测：maxseq64约+.67%、严格TopP16K与maxseq64合计+1.03%、预算16384约+.87%、采样融合+.33%，流式间隔4在旧profile约+.53%。这些结果不能相加，也不证明组合质量。本方向独立新鲜对照组合效应。

控制：CV/KV16/S8、budget4096/83warm、maxseq256、nativeTopP8192/4096、原采样、stream_interval1。候选：相同attention/prefill源码，budget16384/131warm、maxseq64、已通过严格micro的TopP修复16384/8192、固定噪声已检查的采样融合、stream_interval4。两边均async=True、BF16、temperature1.0/top_p.95、4K输入/1K输出/C64/N64。这里16384是预算/内部tile，不运行16K输入benchmark。

同一controller支持3+3筛选（排首轮）和符合>=2%且稳态不交叠后独立5+5（排前两轮）。冻结23关键源码，核验真实预算/序列数/83或131预热/TopP生成源码与compiled metadata/sampler命中/完整流式命令/KV容量及零preemption。两份安装采样源在服务前后核对并原样快照；离线重渲染TopP，与原始响应重算一起验证。

CPU真实执行整条hook组合，两边只组合一次base，保留CV，替换TopP，候选才安装采样融合，恢复runpy.run_path；GPU尚未实测。支持3/5轮的真实嵌套启动参数已检查。本候选不复用任一旧Math；显著且稳定的组合需独立新Math严格>95%，再跑原官方4K。当前async开关实验仍独占GPU，本候选未启动。
