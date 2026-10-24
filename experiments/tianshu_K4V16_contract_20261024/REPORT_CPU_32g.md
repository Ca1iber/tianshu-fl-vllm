# K4/V16：仅CPU契约通过，未接引擎

新方案固定每token/head 128维，K用对称4bit和FP32 scale，V保持原BF16。使用显式RN32 scale/倒数、RN32乘法再RNE，避免把除法与倒数乘法当作逐位等价；小于2^-100整行归零、符号零合并、非有限值不在支持域。原模型权重与当前BM8/r5服务未变。

65280种有限BF16位模式按排序和确定性置换各一次，独立逐行标量FP32参考共1020行/130560元素，全部scale字节与K打包字节精确一致。全部256个packed byte回环、正负半整数ties-to-even、零/极小值和页面K/scale/V区域不重叠通过。原始fixture保存在容器SHM，路径/SHA及源冻结见CPU_GATE_32g.json。

排序族非极小行的平均/最大相对L2约5.381%；置换族平均5.969%、最大6.757%。这两族是编码位域测试，不是实际模型K分布，不代表模型正确率。FP32 scale、原BF16 V、128B padding下拟议页面10496B，相较12288B理论少14.5833%；真实FullAttentionSpec/allocator、实际显存、GPU数值和吞吐全部尚未验证，不能据此声称省下可用KV容量。

下一步需要新GPU writer与K4 QK后缩放读取器，与真正现行解码源同场比较，验证全storage/负slot/scratch/RNG/变数据Graph和数值参考，之后再决定是否接服务、完整Math和4K。BM8当前质量和性能链优先，此CPU准备未插入GPU性能计时。
