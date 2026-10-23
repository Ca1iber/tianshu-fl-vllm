# Down64 W4 v2：独立CPU质量复核

不重新执行GPU、不修改原始记录/源码/模型。重新解析每个batch105份review及prediction/官方report，验证唯一ID、全部105分母、错误与cap计入、官方阈值严格>95%，所有原始文件SHA与源/控制器快照SHA；重新执行全部原有GU/KV/scheduler/sampling/LM、原attention84、原Down除M64 BF16以外所有检查以及新42个W4 capture和实际内存探测/父op绑定检查，完整profile与accuracy_audit逐字相等。

仅C64完成时可生成C64门禁，明确不宣称C8已通过或模型合格。只有两个新鲜batch都>95%才标记model_quality_qualified；独立结果不覆盖GPU原始结果，不替同一失败模型重跑追分。模型checkpoint保持SHA14fb8...；官方serve13 argv及63环境一致，保留903父源。仅CPUimport，不导入torch/triton/vllm/flag_gems。原始报告只在容器，代码与关键分数JSON/CSV入GitHub。
