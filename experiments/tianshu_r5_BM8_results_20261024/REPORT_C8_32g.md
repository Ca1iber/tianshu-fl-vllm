# r5+BM8新C8：100/105，95.2381%，完整通过

新API实际original serve argv，1023来源（原r5全部997源保持）、67显式环境，完整Level3 105题、temperature1/top_p0.95/max_tokens32768。已完成控制器和独立collect_quality_cpu复核：100/105=95.238095%，>95%；全部105题和2次max_tokens都计入分母、请求错误0，stop103/max_tokens2。错误题编号及逐题关键分数见C8_QUALITY_GATE_32g.json和C8_KEY_SCORES_32g.csv；完整原始预测、报告、输入/源码快照、日志在容器持久tar.xz，SHA/字节数和tar比较见C8_RAW_ARCHIVE_32g.txt。

本次确实安装并捕获BM8全部42层，最终JIT/严格机器码/0spill验证通过；但C8运行不产生64路解码请求，不能把启动capture当作真实C64重放证据，C64仍需另起新API验证。原r5全部84个fitted QKV/O权重bank、Down/LM/GU、软件K8/V BF16、内存计数和原始采样规则均重核。较前一次r5 C8 102/105的分差来自非同场配对随机评测，不能据此归因BM8或声称质量完全无变化；本轮只按预定>95%质量门槛判定。

C8控制器33233已正常完成并释放GPU；GPU空闲时迁移纯FX pickle缓存到SHM，旧绝对访问路径保留为symlink，原997源/384固定二进制缓存前后SHA通过，未改模型数值来源。恢复同一个Bash33228后其独立收集C8证据并启动C64，server37142。未重跑C8，未启性能测试。正式r5 6123.31 total tok/s保持不动。
