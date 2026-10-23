# 当前真实Down64 W4：自写Triton解包/反量化/dot组件

只做组件，M64 K6144 N2048，复用已验证42层真实原BF16权重和现有group128对称RTN W4 bank、3个真实GU派生H fixture（各层按行roll，非每层线上激活）、3层×3phase完整FP64参考，不新造权重/不重建量化bank、不改模型或其他冻结文件。对照包含已采用r3实际_iluvatar.ops.mm BF16和前轮已验证native dotsaddz W4，不能只比落后方案。

32个预声明配置：BM32/64×BN64/128×BK64/128×直接重复word load/先load压缩word再expand×一般q*S+Z/对称(q-8)*S，固定warps4/stages1。scale/zero按128-group外提，输入BF16、反量化权重BF16、FP32累加。SYMMETRIC只在CPU先对全部42层实际S/Z和所有16种q证明BF16结果逐字节相同时允许；保留Z原数据。

全部候选先编译、每一项编译资源失败单独保留；非编译设备错误立即停止。全42层×3phase eager与清NaN后Graph重放必须逐字节一致，有限，所有输入/权重/scale/zero全字节/版本/guard/RNG/checkpoint/source不变。原9份完整FP64参考逐格abs误差<=.012*abs(ref)+.006*RMS且relativeL2<=.006，阈值沿用并在新GPU前冻结，不因失败放宽。数值失败保留且不能采用。

计时每次轮换42份真实bank，不反复只读同一权重；graph/eager各7轮、排除首轮、后6轮中位数，各10次42层全调用，轮次按循环旋转再交替反转，保存全部配置结果及实际寄存器/spill/共享内存/IR与二进制SHA。custom复用预分配out，graph为主要结论，eager含不同分配/调度方式不可当模型收益。top性能选择须明确best-of-32，有实质组件收益后另设计独立确认/模型新质量，不直接宣称整机提升或采用。不跑完整benchmark/16K。

代码/报告/关键结果入天数GitHub，原始/数组/编译文件仅容器。最迟UTC2026-10-23T15:58停止，GPU probe内部900秒上限。
