# Down64 + LM64 W4新鲜短4K对照

运行前必须同934来源的新鲜完整C8/C64均>95%，独立105题、raw/CPU bank SHA及全部运行审计通过。原r3精确903来源对候选934，官方13项serve argv/63环境仅PYTHONPATH不同，不能混用之前单Down质量。CPU预检可在质量未结束时check_quality=False；实际GPU prepare默认必须验证完整独立质量门禁，不能绕过。

4K输入/1K输出、并发64、每轮64请求，3轮排除首轮。相同输入seed和8个独立热身/固定canary，所有192请求成功且usage完整，零运行期autotune。继承所有GU/KV/scheduler/sampling/attention/其他M Down与LM8检查，新42W4 Down捕获链接主模型B64真实图执行，LM64为主模型图外真实native W4 calls>=100且旧W8 LM64记录为空。不制造父记录或以Python计数冒充图重放。

固定筛选：候选hot total均值相对精确r3 >0.5%，candidate hot spread<1%，mean TTFT<=control*1.01。CPU准备全部完成后才计时；两臂所有计时期间不运行后台CPU预检/打包/提交/去重，只做轻量状态观察。保留所有原始结果，不原样重跑追分。不跑16K。最迟UTC2026-10-23T15:58停止。
