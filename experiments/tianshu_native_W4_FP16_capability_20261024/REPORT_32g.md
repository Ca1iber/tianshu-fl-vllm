# Native W4 FP16：真实Down接口及数值资格通过，尚未计时

2026-10-24，BI-V150，PID29498已正常结束。原生dotsaddz实际接受原layer0 Down的FP16 A/Scale/Zero、自然顺序原W4 pack，M64/K6144/N2048/G128，输出FP16。测试没有修改checkpoint、模型、引擎或r5。

当前BF16 W4与新FP16 W4各三组真实激活，完整786432个输出与各自独立CPUFP64解码矩阵投影核验通过，重复eager和同Graph变化输入、NaN毒化输出重放逐字相同，所有输入/pack/scale/zero字节、version、guard及CPU/GPU RNG保持。FP16相对自身矩阵的L2误差约0.000207，BF16约0.00166；这包含输出dtype差异，不能当作模型正确率改善。

本bank的scale/zero从BF16转FP16逐数值无损；三相位激活分别44/152/150个极小值发生变化，最大绝对差2.9802322387695312e-8。解码权重的FP16与BF16舍入不同，参考单独建立，不宣称通用逐位等价或其它bank也已验证。原文件和3份SDK摘要前后匹配旧门禁；没有复制大SDK库。

本阶段没有任何计时、服务、Math或benchmark，performance_qualified=false。下一步以42个当前r5的真实DownW4 bank比较原BF16、裸FP16和含BF16→FP16→BF16完整转换的Graph，再判断是否值得更深融合或模型验证。原始输出与FP64参考已在容器持久归档，两个archive逐文件tar compare通过，SHA/位置/大小见RAW_ARCHIVE；GitHub只存源码/本报告/关键门禁。
