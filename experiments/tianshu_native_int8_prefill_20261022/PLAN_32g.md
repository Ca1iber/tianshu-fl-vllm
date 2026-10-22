# 原生 INT8 GateUp 的大M预填充完整路径

M64原生完整路径数值全位相同但约0.066ms慢于现用0.063ms，不能推广到大M。新实验只测当前已采用W8A8融合的M128/512/4096，连续实际K2048,N12288；从正式range_policy及selected_plan提取每M原生控制tile，并严格比对actual_timed_candidate_kernels二进制。原始BF16 checkpoint与W8 bank签名保持，不改M白名单/架构/量化方法。

每M当前量化+Triton两dot融合，对照同一量化+torch._int_mm(out)+scale/BF16投影/同一FlagGems SiLU epilogue。候选2布局(NN/NT)×2种epilogue行分组(1或4rows，BLOCK1024/W8)，预先固定4候选。全量INT32和BF16投影/激活位一致，3相位修改A和W/scale，同Graph重放；guards、输入、RNG不变，7轮轮换×20replay，所有运行时中间读写计入。

独立CPU重建checkpoint全部weight格式和A8、选中INT64 dot，依据完整保存的INT32输出与scale计算全BF16投影hash，实际GPU每候选的projection/output完整hash与当前控制比较，另存selected activation以做FP64数学参考。每相位只存一次共同INT32，不重复保存各候选相同大数组；大数组和日志只在容器，GitHub仅代码/计划/关键JSON/CSV/报告。

最大INT32单workspace201326592字节，模型接入的Graph生命周期、DeviceMemoryProfiler与bank布局成本尚未审核。真实权重只读，前后SHA一致。组件阳性才考虑全42bank holdout和服务小样本；正式收益必须fresh C8/C64 Math与官方4K。组件不借旧质量，不以中间GEMM计时冒充全路径。只用天数，06:00–24:00授权、23:58停止，单GPU任务最多30分钟；冻结后修复需新版本。
