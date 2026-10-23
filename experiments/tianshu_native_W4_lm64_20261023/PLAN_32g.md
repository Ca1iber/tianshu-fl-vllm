# 真实 LM Head64：W4原生两条路径对当前W8A8核心

权重为原始checkpoint lm_head.weight [130560,2048]，原始文件不改。当前r3实际使用 b9b692 的dense.quantize + dense.mm，配置bm64/bn128/bk128/warps4/stages1（不是前版未采用的原生_int_mm/dequant），故以此为对照。r3正式记录显示LM64全部15385次为图外调用，eager和graph均测、明确标注组件核心范围。

W4 group128 RTN，先量化scale到BF16，再以实际scale进行round/clamp[-8,7]+8，S [16,130560]；AWQ使用已验证N偶数先于奇数pack和packed零8，dotsaddz使用已验证自然N打包和BF16加性零-8*S。两候选逻辑量化权重相同，未做激活感知校准。完整267386880个逻辑uint4打包均CPU独立逆变换精确，当前W8 codes/scales完整SHA须与已通过模型的pack签名一致。

3个独立确定种子的BF16 M64随机fixture，每个包含一行零，且保存全量；不是模型在线activation。CPU预先计算每phase完整64x130560原始BF16/反量化BF16 FP64参考。当前W8A8使用完整A8/scales独立重算及全64行x512等距列（含尾列）的全K INT64、确切BF16顺序参考。W4所有输出逐格误差<=.012*abs(ref)+.006*RMS(ref)且相对L2<=.006，全部有限；阈值首GPU前固定，量化误差相对原始投影另报。

GPU对3个variant、3phase保存所有输出并检查eager/graph poison重放逐字节一致、输入/权重/scale/zero完整字节/版本/guard、RNG。对照可复用预分配q/scale/output核心缓冲（偏向对照，明确不是整个模型LM调用的CPU开销）；候选含原生输出分配。核对当前W8 compiled的93regs/0spills/24576shared，903来源和库SHA。

性能各7轮交替/旋转顺序，每轮20次；首轮排除，分别取其余6轮中位数。单份W4权重133MB已远大于小组件热bank，报告存储和实际矩阵尺寸。仅组件，不改变服务/模型架构，不跑正式benchmark；只有速度/数值均通过才考虑新鲜Math500与服务集成。单GPU进程900秒上限、23:58北京时间前结束，代码/报告/关键结果GitHub，大数组仅容器。
