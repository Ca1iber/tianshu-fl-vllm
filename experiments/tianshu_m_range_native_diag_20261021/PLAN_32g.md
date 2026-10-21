# O/M8原生对照诊断

原BF16范围计划在O8第1相位停下，保留failed与51项已完成证据，不改旧源，不扩大.02+.01数值阈值。新诊断使用同一原模型O权重、同seed BF16输入，保存完整A/W和所有实际输出，FP64 CPU全矩阵参考，区分固定pointer Graph、重复1/20/140、同pointer eager、新pointer eager/Graph。

原生公共API实查为_iluvatar.ops.mm，非通用库误用。本进程只在Python包装实际mm_kernel返回值，原生设备函数与参数原样转发，收集真实CompiledKernel常量/二进制/TTIR，不改已安装source或模型。既定数值差异作为诊断数据记录，不将diagnostic complete冒充原组件通过。先确认是否SPLIT_K与BF16 partial atomic、缓存或Graph行为，再设计下一轮完整对照。

GPU空闲后执行，最晚23:58北京，最多300s。本诊断不测吞吐、不运行模型服务。所有源/CPU gate/计划先冻结，修改需新目录。
