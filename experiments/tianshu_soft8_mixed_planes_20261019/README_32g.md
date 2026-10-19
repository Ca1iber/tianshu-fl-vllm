# K/V分别存储精度：仅组件准备

当前软件8位K/V已通过原命令正式4K4259.87及新Math102/105。此处新假设是：只保留K或V为BF16，另一面仍按同一软件8位格式存储，增加的访存量也许能换取更少解码运算/寄存器。不能预先认定有收益，也不沿用当前质量证明新布局。

10种batch/长度、BF16/全8位原writer/全8位自有writer/K16V8/K8V16五路（含当前原写入与自有写入的全8位对照）；混合页实际12288bytes，使用真实FullAttentionSpec/page_size_padded和原生reshape获得原始缓存view，两种typed plane与同一raw storage别名。此组件直接调用writer和unified decode，不声称真实ModelRunner已安装混合布局，也不测试prefill。

GPU计划穷举65536实际writer编码、256解码值，并逐物理字节检查两次改变输入/slots/长度的同一Graph、padding/输出/scratch/input/RNG，针对每种布局独立FP32和CPU参考。事件包括writer+reader，当前byte的分段/launch策略固定，比较每次仅存储类型变化。若不优于当前byte则不接入框架；若有收益还须完成真实allocator生命周期、prefill、模型及新Math/正式验证。
