# 独立CPU汇总新C8与真实C64质量

从当前两轮原始prediction/review/report重新计算，而不是只读通过flag。C8 raw原audit_failed不改，独立复核按实际BF16预填充范围明确保留101/105、INT8运行命中0；C64必须原GPU完整审计complete、全105新题>=100正确、实际M4096原生INT8全42层、M64图和LM64 INT8覆盖。两臂同877模型源、原471精确子集、原13 serve argv、实际63环境；C64仅准入候选转原生control，配置差异严格记录。

所有模型源和控制器按实际source snapshot重校hash；重跑当前范围的完整runtime审计与原始accuracy算法，检查请求、缺失、错误、停止原因和全部105索引。只上传精简gate/报告/关键原始报告hash，完整profile、预测和日志留容器。正式4K仍在运行，不把已通过质量或短测当作新的正式成绩。
