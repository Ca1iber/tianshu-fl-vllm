# LM小批次首版M4 BF16对照图一致性失败

冻结8df8b37。M1/M2各3phase eager/Graph输出一致、全部计时完成；M4 phase0的`BF16_actual_embedding_apply`检查失败，程序立即退出，未执行后续M，未运行完整独立CPU审核，因此本版不报告已验证完整性能收益。原始失败及之前所有输出、逐轮计时保留，不能作为模型质量或正式成绩。

该assert同时检查Graph/eager字节一致及finite，原版未在assert前保存M4失败输出，因此现有证据不足以区分数值差异、signed-zero或NaN，不能推断根因。后续新冻结诊断将M4的每次eager、未poison graph以及poison graph输出全部先保存，再独立FP64核对；不修改原版或放宽其门槛。真实对照warm调用已确认进入vendor linear，并非拿显式mm冒充实际LM apply。

原始`/workspace/benchmark-results/20261023_native_W4_lm_small_GPU_tianshu/result.json`。已采用r3和模型文件不变，无新Math/正式benchmark/采用。
