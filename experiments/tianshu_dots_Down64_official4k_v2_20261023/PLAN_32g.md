# 单一Down64 W4 v2：有条件正式4K控制器

只在同模型新鲜C8/C64完整独立质量门禁均>95%且新短4K配对筛选>0.5%、candidate spread<1%、TTFT<=control1.01后运行。CPU准备可跳过未出炉的short筛选，但真实GPU prepare默认必须验证screen；不存在绕过质量门禁路径。

使用原始官方benchmark_throughput_serve.py原SHA、官方13项serve argv、903父来源完整保留共916来源、最终63环境，测试原始 [[4096,1024,64,256]]，4轮、排除第一轮、后3轮均值。不得改成N64或拿小样本作为正式成绩；无16K。保留全部旧GU/KV/scheduler/sampling/LM/attention、Down除M64 BF16以外检查，新42个W4捕获与actual B64模型图执行链接、真实模型内存/父op/schema/bank绑定必须通过。

对原正式r3 5950.75 total /1190.15 output tok/s、TTFT3174.58ms的比较明确为历史成绩比较，不冒充同一次完整AB；另附新鲜配对short门禁与SHA。晋级要求相对r3>0.5%、后三轮spread<1%、TTFT<=3174.58*1.01，同时新鲜质量双通过、所有运行审计、独立CPU正式结果复算通过。失败保留不原样重跑追分、不改源/参考/阈值。最迟2026-10-23T15:58Z结束。代码/报告/关键CSV/门禁入GitHub，原始全部在容器。
