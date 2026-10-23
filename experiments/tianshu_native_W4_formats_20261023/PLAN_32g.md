# 原生W4非恒定矩阵的打包/零点格式探测

依据实际Torch Meta/调度、二进制输入检查和CoreX4.5的ixinfer.h：原生AWQ接受BF16/Half激活及int32二维weight，group128合法，scale/zero为二维；libcuinfer提供W4A16 AWQ/GPTQ类型。具体零点/打包尚不据此声称已确认，以下全部预先声明为待检验假设。

同一个64×256×256非恒定逻辑矩阵，X有正负/不同行，q随K/N变化、两K组的scale和zero也随列改变。全部16384个输出用完整K的INT64投影再除128与独立BF16舍入计算；原生CPU通用INT4使用自己的转换器完整核对。每个物理pack在CPU逐格逆解码，Meta形状单独检查，不冒称GPU硬件支持。

2个通用GPU uint8 K向pack假设（偶数K在低/高nibble），仅前轮通用微测试通过才执行；6个AWQ格式假设为N向natural/[0,2,4,6,1,3,5,7]与packed整数zero/BF16数值zero/BF16加性offset交叉。逻辑权重与参考始终相同，错误格式的真实失败/不一致保留，不改预期。scale/zero/输入额外初始化存储且逐字节/版本守卫，8个变体各独立进程45秒上限，不计时、不改模型。

首次GPU前冻结代码/逻辑输入/物理输入/实际库SHA。等待PID43163结束，GPU空闲且距15:58 UTC至少15分钟才执行。通过仅表示该小矩阵的实际格式与原生接口正确；真实模型weight精度、性能、Math正确率均仍需后续独立实验，不自动接入模型或完整benchmark。
