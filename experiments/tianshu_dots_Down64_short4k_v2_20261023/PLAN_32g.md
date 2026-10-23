# r3 vs 单一Down64 W4 v2：新鲜短4K配对筛选

对照为精确r3 903来源，候选为冻结998971f的916来源（保留903父来源，仅Down64运行时RTNg128 W4 dotsaddz）。13官方serve argv保持、最终63环境仅PYTHONPATH实现入口不同。实际GPU前必须完整同版本新鲜C8/C64均>95%、105题分母及完整独立CPU门禁合格；CPU准备可以check_quality=False，但真实run默认True，不允许绕过。

沿用已冻结的普通N64测试器：4096输入/1024输出/并发64/64请求，每臂3轮，首轮排除，后2轮均值；相同输入seed列表，完整每条stream response usage5120与TTFT记录；seeded canary只要求各臂自己重复一致，不要求新精度与旧精度逐字相等。每轮无新增Triton tuning记录。控制/候选串行，GPU无其他作业，维护真实KV/scheduler/GU/attention/LM全部原审计，候选只替换Down64 BF16捕获检查为42个真实W4 capture和actual_B64_capture_and_model_replay_verified链接。

仅候选相对对照均值>0.5%、候选热轮spread<1%、TTFT<=对照1.01才通过筛选，独立CPU重算全部记录/源与快照/实际profile/权重checkpoint后落门禁。首轮不得挑选重新定义，失败不原样重跑追分；结果不当作正式N256成绩。无16K、无自动正式benchmark。北京时间23:58前停，所有源码先于GPU冻结；大数组/原始stream只容器，GitHub代码/关键CSV/门禁/报告。
