# 逐token/head K尺度与INT8 QK：仅CPU格式准备

实际Corex INT8 dot能力已通过，但尚无完整attention或性能结果。新格式探索将K按每token/head最大绝对值存signed int8及FP32尺度，Q在解码线程块中按query/head量化后直接整数QK乘法，再乘两侧尺度；V继续BF16。目标是省去每个K tile解码成BF16再做QK乘法的路径。这里只定义K格式，没有GPU attention、服务或Math，未采用。

物理页12416字节：K4096、同现有布局的BF16 V8192、32个FP32 K尺度128字节。CPU证明全部字节不重叠且恰好覆盖页面。有限BF16输入契约，尺度max(abs,1e-8)/127，FP32 nearest-even算术，signed码限制[-127,127]，128字节码保留；解码乘FP32尺度后BF16 nearest-even，零码归+0。

独立struct IEEE FP32计算及BF16位舍入参考检查全部65280有限BF16位模式、两次重新分组和零/半整数ties/最大值，共196352个标量；编码、FP32尺度和解码位逐项一致，输入不变。这个契约检查不证明模型质量，也不把INT8 dot原型的通过当作该新格式已经接上真实引擎。后续必须经过writer/分页/混合长短序列、同Graph变数据、服务与全新Math严格>95%。
