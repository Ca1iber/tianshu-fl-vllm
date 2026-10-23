# r4 中性温度同次短4K筛选

前置actual Sampler完整缩放bits/token/default与逐请求RNG均通过，CPU重核全部记录来源SHA。两组均原r4的944源、13项原命令、同一63键原配置（PYTHONPATH仅组合新增观察/策略），另加相同mode和record路径两个键；区别仅mode control/candidate。control仍做原除法，candidate仅实际CPU metadata全温度1时跳过。3轮64请求、4K/1K、并发64，剔除首轮，保留完整响应和tuning/metrics/canaries。此次仅筛选，不借用旧Math当新优化资格，不宣称已通过完整继承正式审计；正式发布仍r4。全源前后SHA与实际env、路由、token用量、冷salt canary及GPU空闲检查保留。两组原始输出路径固定，异常不覆盖。硬截止23:58。

本v2双方使用单位温度v2观察器定期落盘；唯一计算区别仍mode。原控制audit_failed保持，不使用其数字替代本次新对照；其余用例、阈值、请求数、剔除规则、原944源与13argv不改。
