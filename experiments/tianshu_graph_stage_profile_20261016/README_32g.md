# 天数：原始命令下的 Graph 与框架阶段诊断

本轮已完成、备份并通过本机离线核验。真实模型没有启用 IXsys，稳定 C64/FULL Graph 的 40 步中，模型前向约占流区间的 88.48%，词表投影 2.10%，采样 9.16%。总区间中位 54.179 ms，前向 47.932 ms、词表投影 1.138 ms、采样 5.034 ms。这是诊断，不能换算为正式吞吐或当作单个 kernel 的活动时间。当前正式 4K 成绩仍为 3992.97 total tok/s，Math 103/105。

IXsys 子进程覆盖计数已实测，但完整能力检查失败：V1 20 Graph/20 图外 kernel，V2 25 Graph/20 图外 kernel；两版均未看到图内 kernel。V2 无采集图外事件 10 回放/1 回放约 9.84 倍；无采集单图 0.119873 ms，而开启采集后的 trace 中位 0.426340 ms，原定 .5..1.5 校准门槛未通过。没有放宽门槛，也没有将失败结果标记通过。

真实模型诊断保持官方原始 serve argv、默认 2048 tokens/256 sequences、75 预热点、CV/KV16/S8/nativeTopP/原采样器。hooks/sitecustomize.py 包裹实际 ModelRunnerFL 路径，arm 后先等 32 步连续 C64 decode，再采 40 步；事件都在 Graph 外，每步无同步，窗口两端各一次同步。前后 API seed777 canary 一致，128 条负载响应、50 份源码快照、实际 39 个环境项及零 preemption 已离线核验。

完整报告和重放命令见 [REPORT_32g.md](results/20261016/profile_32g/REPORT_32g.md)，机器记录为同目录的 CAPABILITY_AUDIT_32g.json、MODEL_STAGE_AUDIT_32g.json。未改最佳启动器，服务和 GPU 任务已退出，未排队下一轮实验。
