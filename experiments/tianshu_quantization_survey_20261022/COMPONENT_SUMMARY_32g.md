# 天数权重量化组件汇总，2026-10-22

原checkpoint只读，未转换架构、未微调。比较BF16 activation下的4/8bit权重存储及实际解包/GEMM成本；模型原文件前后SHA保持。全部候选目前未接模型、未跑新Math/正式benchmark；原正式发布5695.95 total tok/s保留。

|独立组件版本|Down/M64 最快格式、ms|LM/M8 最快格式、ms|LM/M64 最快格式、ms|
|---|---|---|---|
|逐元素解包|W8_channel / 0.350848|W8_channel / 1.885439|W8_channel / 4.314962|
|成对读取解包|W8_channel / 0.350654|W8_channel / 1.884101|W8_channel / 4.268631|
|W8 NN后缩放|W8_channel / 0.338771|W8_channel / 1.877103|W8_channel / 4.094505|
|W4 NN分组后缩放|W4_sym128 / 0.462430|W4_aff128 / 2.923444|W4_aff128 / 5.673891|
|W8 NT后缩放|W8_channel / 0.226201|W8_channel / 0.982689|W8_channel / 2.779977|
|W4 NT分组后缩放|W4_sym128 / 0.209211|W4_sym128 / 0.898102|W4_sym128 / 2.371455|

每点为3个预定tile的7轮×20次热Graph中位数中最小值；完整格式/tile数据在各目录KEY_TIMINGS。当前已采用Down/LM8的BF16、LM64的W8A8，不能用较慢BF16替代已采用的LM64控制来宣称收益。activation为合成BF16，weight来自实际checkpoint第0层Down/LMhead；并非完整模型benchmark。

全部列出的组件候选均未超过对应现用路径。结论局限于这些实现、shape和tile；不能推断天数所有INT4实现均不可加速。成对读取、NT布局确实减少了原型成本，但仍有显著差距。已安排IXKN实际kernel计数器定位，计数器采集结果另行报告。

格式载荷（包括FP32 scale/uint8 zero）为BF16的26.6%–28.9%，与引擎显存净节省不同；保留BF16 fallback时属于新增副本。当前已采用INT8 GateUp/LM64及K8/V16 cache的质量/吞吐证据不借给新路径。

6版均验证尾部/stridedA、3次变更同一图输入、完整format decode位级、全输出FP32/BF16 cast、完整GPU参考、独立CPU FP64抽查、所有A8量化和selected W8A8整数结果、guards/RNG/输入不变。原V1 signed-zero测试参考错误与离线V1序列化错误另存，不覆盖失败记录。

GPU每日北京时间06:00–次日00:00，本日23:58截止。完整数组和profile仅容器，GitHub保留代码、报告和关键结果。
