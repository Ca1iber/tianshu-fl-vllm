# 天数 BI-V150 32g：五项小收益的组合候选

当前正式最好4091.29 total tok/s、Math103/105。单项服务短测：maxseq64约+.67%、严格TopP16K与maxseq64合计+1.03%、预算16384约+.87%、采样融合+.33%，流式间隔4在旧profile约+.53%。这些结果不能相加，也不证明组合质量。本方向独立新鲜对照组合效应。

控制：CV/KV16/S8、budget4096/83warm、maxseq256、nativeTopP8192/4096、原采样、stream_interval1。候选：相同attention/prefill源码，budget16384/131warm、maxseq64、已通过严格micro的TopP修复16384/8192、固定噪声已检查的采样融合、stream_interval4。两边均async=True、BF16、temperature1.0/top_p.95、4K输入/1K输出/C64/N64。这里16384是预算/内部tile，不运行16K输入benchmark。

同一controller支持3+3筛选（排首轮）和符合>=2%且稳态不交叠后独立5+5（排前两轮）。冻结23关键源码，核验真实预算/序列数/83或131预热/TopP生成源码与compiled metadata/sampler命中/完整流式命令/KV容量及零preemption。两份安装采样源在服务前后核对并原样快照；离线重渲染TopP，与原始响应重算一起验证。

CPU真实执行整条hook组合及3/5轮嵌套CLI已通过。2026-10-16新鲜3+3筛选4060.1406→4160.3683 total tok/s，+2.468579%，稳态TTFT约+5.50%；384响应/23源已全本地审计，详细报告见results/20261016/screen3_32g。达到预设门槛后启动独立5+5确认，预排前两轮。尚未采用、不复用旧Math；独立确认通过后需新Math严格>95%，再跑原官方4K。

独立5+5完成：后三轮4065.8366→4149.6193，+2.060652%，TTFT增加约6.1%，640响应/23源全本地审计，10轮无新增调优。详见results/20261016/confirm5_32g。新Math/正式尚未启动；新查公开规则要求serve参数与baseline一致，当前额外参数的复现条件待对齐，不据此宣布新比赛最好。
