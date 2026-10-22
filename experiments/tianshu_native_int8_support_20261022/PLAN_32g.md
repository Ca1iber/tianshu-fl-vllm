# CoreX 原生INT8 GEMM接口支持性检查

已安装torch2.10.0对aten::_int_mm和out注册CUDA kernel，CUDABlas.h声明int8_gemm；这仅证明接口存在，不能推断BI-V150运行或性能。历史报告未发现此接口试验。本次仅验证真实GPU支持、NN/NT输入布局与CUDA Graph out捕获/变化输入，先不读取模型，不计时、不宣称模型速度/质量。后续若支持，需真实GateUp bank与现版融合kernel完整对照，计入A8、INT32中间写读、缩放及SiLU成本。

预定M8/64,K2048以及M64/K6144，N256；CPU构造整数输入，3相位改变A和W，完整CPU INT64点积验证原生INT32，守卫/输入/RNG保持。布局NN contiguous和NT-transposed两种。源/计划先冻结，失败保留；GPU前检查空闲，单任务120秒、今日23:58硬停。无其他GPU实验同时运行。全部文件只在容器，代码报告关键结果推天数GitHub。
