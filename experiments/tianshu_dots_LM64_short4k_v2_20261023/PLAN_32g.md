# Down64 + LM64 W4新鲜短4K对照

运行前必须同934来源的新鲜完整C8/C64均>95%，独立105题、raw/CPU bank SHA及全部运行审计通过。原r3精确903来源对候选934，官方13项serve argv/63环境仅PYTHONPATH不同，不能混用之前单Down质量。CPU预检可在质量未结束时check_quality=False；实际GPU prepare默认必须验证完整独立质量门禁，不能绕过。

4K输入/1K输出、并发64、每轮64请求，3轮排除首轮。相同输入seed和8个独立热身/固定canary，所有192请求成功且usage完整，零运行期autotune。继承所有GU/KV/scheduler/sampling/attention/其他M Down与LM8检查，新42W4 Down捕获链接主模型B64真实图执行，LM64为主模型图外真实native W4 calls>=100且旧W8 LM64记录为空。不制造父记录或以Python计数冒充图重放。

固定筛选：候选hot total均值相对精确r3 >0.5%，candidate hot spread<1%，mean TTFT<=control*1.01。CPU准备全部完成后才计时；两臂所有计时期间不运行后台CPU预检/打包/提交/去重，只做轻量状态观察。保留所有原始结果，不原样重跑追分。不跑16K。最迟UTC2026-10-23T15:58停止。

v2仅修正执行前质量门禁：C64原934/95模型105题完成101正确，原审计INT8期望遗漏，完整CPU独立校正已通过，原result仍audit_failed；要求新完整QUALITY_GATE显式绑定CPU校正与原失败SHA，C8必须原版本正常complete。不重跑C64，模型数值、N64×3协议、所有计时阈值不变。v1从未GPU执行。CPU预检复用已验证的同903/934来源快照，逐文件SHA重核，并核对当前同934 C8真实13argv/63环境，不复制大快照；不把部分C64资格当双资格。
