# K8V16 Decode launch sweep（仅CPU准备）

当前K8V16真实服务筛选比双8位约+17.24%，全新Math100/105已通过，完整4K正在运行；本目录是后续独立假设，不改变其196份源，也不并发用GPU。K8V16寄存器142，旧双8位245，旧T16/w1/分段策略未必仍为新布局最优。

预设11配置：保留当前policy control，扫描T16/32/64、w1/2/4、S2/4/8/16和一组stage2。9种B/ctx（含B64/33短上下文，覆盖候选Graph预热分段边界），仅3D Decode；所有路径实际同12KiB K8V16页面、相同writer/codec、相同逻辑Q/K/V和完全相同页表值。分段scratch足额64，输出/物理存储/输入/padding/RNG及变数据同Graph检查保持，每种配置独立FP32与CPU参考。7×20 event顺序轮转并反转、记录实际顺序。组件不是服务或质量成绩，任何新配置采用都需独立验证。

CPU已执行198组实际Producer分支和writer encoder AST同一性。尚未GPU运行/源码编译资源验证。必须等当前Math、正式benchmark终态并释放GPU，再决定运行本组件；不要为此中断它们。
