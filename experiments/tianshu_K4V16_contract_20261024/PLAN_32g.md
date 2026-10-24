# 新研究方向：按token/head缩放的K4，V保持BF16

只定义独立CPU数值/打包契约，不改r5/BM8服务或checkpoint，不启动GPU。现有V8多次负收益；K4的待验证区别是K逐token/head缩放可在QK点积后按token缩放score，不必逐V特征反量化。能否因此更快尚无证据。

支持有限BF16键值、每token/head 128维。maxabs<2^-100的整行编码为0并scale=1，以排除倒数溢出；其它行scale=RN32(maxabs/7)，inv=RN32(1/scale)，q=clip(RNE(RN32(x*inv)),-7,7)。q+8存4bit，偶D为低nibble、奇D为高nibble，0码保留；符号零统一编码8，非有限输入不在本契约支持域。scale为FP32，避免把BF16 scale舍入与K4一起引入。

拟议页面block16/2KVhead/D128：K数据2048B，scale128B，padding128B，V BF16 8192B，合计10496B。相对当前12288B只是理论布局减少14.5833%，尚未验证真实FullAttentionSpec、allocator或引擎容量，不得把它当作已释放显存。原V BF16全部位保持，模型参数/结构不改。

CPU先穷举65280种有限BF16位模式（排序和确定性置换各一次），独立逐行标量FP32舍入参考、全部256个打包byte、RNE半整数边界、符号/零/极值与页面区域不重叠。GPU后续必须比较真正现行r5读取器（若BM8已正式采用则以BM8为父源），同一原始Q/K/V输入，完整实际writer/cache/输出/scratch/负slot/RNG保护、同Graph原地变数据、FP32参考/FP64重算及真实Spec计数。QK后缩放与prefill BF16反量化的舍入差须单列。

只有组件正确且跨4K/5K/strided均有可复现收益后才接模型、重新完整Math>95%、同场4K筛选和必要正式benchmark。CPU契约通过不代表GPU正确、模型质量或端到端提速。BM8的C8/C64/4K验证优先，本研究不插入任何GPU计时。
