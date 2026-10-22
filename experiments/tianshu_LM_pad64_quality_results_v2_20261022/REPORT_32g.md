# LM Head M32..64补齐：完整双批次质量

925个模型输入，原GPU质量控制82项；903个已发布r3模型输入逐字节包含且未改。原checkpoint与模型结构不变，权重文件SHA256=14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c。仅新LM Head M32..64逐行量化补至64并融合输出转换；M<=8及M9..31保留BF16，QKV/O仍为r3策略。

新完整Math-500 Level3，两个独立批次均严格大于95%：

| 并发 | 正确/全部题 | 正确率 |
|---|---:|---:|
|C8|102/105|97.142857%|
|C64|103/105|98.095238%|

每批均105个成功API响应，0 API错误；完整原评审、预测、报告三方逐条核对并独立重计，达到输出上限的题照常计入。KEY_SCORES_32g.csv保留全部210条关键分数与停止原因，不上传题目/预测正文。

C8实际Math阶段未调用新高M LM路径，实际LM8 BF16调用与dtype已核验；C64真实Math阶段M32..64全部命中新路径，计数扣除启动零探针，原QKV/O与Down各42层M4096路径及原M64 BF16策略保持。原903输入及925模型/82GPU控制快照、pack/CUBIN/真实内存分析、checkpoint、实际argv13项和环境63项全部核验。

C64原GPU控制器由于实际Math旧LM64计数器未更新而audit_failed，该原失败永久保留。新CPU恢复使用相同105份结果，不重新生成；先复现旧失败，再仅更正计数来源、保留阈值与全部其他审计，完整重新核验。五项CPU恢复输入、快照与执行轨迹来源独立记录。质量Gate明确C64_original_experiment_passed=false、C64_CPU_runtime_audit_recovered=true；不能声称原C64完整验证器通过。详情见相邻audit_recovery_20261022报告。

独立QUALITY_GATE_32g.json允许进入吞吐筛选，不表示性能正收益、正式最佳或已采用。原始数据、快照与完整日志均在容器/workspace/benchmark-results/20261022_LM_pad64_math{8,64}_tianshu，CPU恢复另在math64_CPU_recovery_tianshu；GitHub只代码、报告和关键结果。
