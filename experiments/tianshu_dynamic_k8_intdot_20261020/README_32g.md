# 逐token/head K尺度与INT8 QK：仅CPU格式准备

实际Corex INT8 dot能力已通过，但尚无完整attention或性能结果。新格式探索将K按每token/head最大绝对值存signed int8及FP32尺度，Q在解码线程块中按query/head量化后直接整数QK乘法，再乘两侧尺度；V继续BF16。目标是省去每个K tile解码成BF16再做QK乘法的路径。这里只定义K格式，没有GPU attention、服务或Math，未采用。

物理页12416字节：K4096、同现有布局的BF16 V8192、32个FP32 K尺度128字节。CPU证明全部字节不重叠且恰好覆盖页面。有限BF16输入契约，尺度max(abs,1e-8)/127，FP32 nearest-even算术，signed码限制[-127,127]，128字节码保留；解码乘FP32尺度后BF16 nearest-even，零码归+0。

独立struct IEEE FP32计算及BF16位舍入参考检查全部65280有限BF16位模式、两次重新分组和零/半整数ties/最大值，共196352个标量；编码、FP32尺度和解码位逐项一致，输入不变。这个契约检查不证明模型质量，也不把INT8 dot原型的通过当作该新格式已经接上真实引擎。后续必须经过writer/分页/混合长短序列、同Graph变数据、服务与全新Math严格>95%。

Reader准备阶段仅修改统一attention kernel：3D时Q逐行量化、K按页尾FP32尺度读取并执行INT32 QK，随后按两侧尺度恢复分数；2D保留Q BF16并将新K按尺度恢复BF16。参数ABI及其余helper/reducer AST不变，修改可精确逆变换；页表读取增加有效token mask、尺度读取沿用decode的.cv。2048项CPU整数缩放代数与2,064,512最坏累加界验证通过。Reader尚未GPU编译，2D/prefill也没有GPU覆盖。

准备11个纯decode场景（B1/8/32/64的4K/5K、两组ragged/stride、B40/16K），与现有软件E4M3 K8V16分开按各自数值格式构造参考。新格式参考的有效Q/K由CPU独立编码及FP32尺度构造；FP32 attention与CPU向量参考，完整K/V/尺度/输出/scratch/RNG及同Graph变数据，计时包括writer。误差门槛atol0.002+rtol0.01，未以旧格式输出逐位相同为前提。只新增shared memory观测字段，归一化后的Producer计算AST与当前launch相同。必须先通过排队中的GPU格式检查及单warp INT8 dot入口，再运行此probe_decode.py；未接真实引擎或Math。
