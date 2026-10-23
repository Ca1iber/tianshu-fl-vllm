# Down64 Triton W4组件：32配置数值全部通过，但明显慢于原生

使用全部42层真实Down权重和已验证group128 W4 bank，对照当前r3实际BF16模块和已验证原生dotsaddz W4。32种预声明tile/解包/对称计算配置全部编译，全部通过独立CPU数值复核：3层×3phase完整FP64参考合计40108032格；全部42层×3phase输出有限、eager与清NaN后的Graph重放逐字节相同；输入、权重、scale、zero全字节/版本/guard、RNG、checkpoint及源码不变。

Graph每轮调用10次全部42层，7轮排首轮取后6轮中位数。BF16约2.320ms，原生W4约1.973ms，自写最快`tl_m64_n64_k128_p1_s0`为13.525959ms，比原生耗时约6.86倍。最快点是best-of-32选择，后6轮spread约0.1413%，不能当独立确认。eager结果同样为负；自写复用输出，分配方式与原生有差异，主要结论采用Graph。

最快点71个寄存器、0spill、32768字节shared。IR和汇编确认存在`llvm.bi.matrix.mad.f32x4.bf16x4`和`ml_matrix_mad_f32_bf16_m16n16k16`，因此不能归因为BF16没有硬件矩阵指令。packed词扩展有布局转换、共享内存读写及barrier，支持继续调查这些开销，但本轮没有profile数据证明唯一瓶颈。所有32点的实际资源、IR、汇编和binary SHA保留在容器；GitHub保留关键JSON/CSV与冻结源码。

组件未接入模型，没有新增Math或正式benchmark。保持已采用r3 5950.75 total /1190.15 output tok/s、TTFT3174.58ms；这不是本次新成绩。原始数据仅容器`/workspace/benchmark-results/20261023_triton_W4_Down64_GPU_tianshu`。源码冻结commit `60f2e03`，独立审核与全部计时见本目录JSON/CSV。此轮不会原样重跑。
