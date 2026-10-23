# keep128新C64：102/105，独立部分核验通过

本版本928模型输入/51质量控制器，原r3的903源逐字节保留，只新增FULL/64token/64req/uniform/noLoRA QKV64原生量化，原QKV128 native保留，不混入LM925方案。新鲜完整Math-500 Level3 C64全部105个API成功，0错误，独立逐题重计102正确（97.142857%），严格>95%。两个达到32768输出上限的样本索引26、83均按原评审完整计分，不剔除。

原完整验证器全部运行审计通过，独立验证再次核对105题原报告/评审/预测、928模型及51控制文件SHA和快照、原checkpoint SHA、实际13argv/63环境、84个真实pack及内存、六次预热/CUBIN、QKV64捕获42层native/O64 BF16、非均匀M64回退及Down42层审计。真实运行QKV native命中M128/4096，O命中4096；QKV/O各42bank实际M4096，M128没有被关闭。Python路由计数不当作图重放次数。

仅C64部分结果，QUALITY_GATE_32g.json的model_quality_qualified=false。新C8仍在运行，不能由此启动benchmark、采用或发布；不得借用任何其他928版本、903/915/925分数。完整双批次结果由相邻quality_results_20261023目录在新C8结束后独立产生。

全部105条关键分数与停止原因在KEY_SCORES_32g.csv；原始文本与日志只在容器/workspace/benchmark-results/20261023_QKV64_keep128_math64_tianshu。使用首GPU前冻结的独立验证器同字节副本--batches64，未重跑GPU或修改原始结果。
