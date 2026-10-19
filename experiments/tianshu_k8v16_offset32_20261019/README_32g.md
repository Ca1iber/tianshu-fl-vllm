# K8V16 地址偏移32：独立候选准备

当前只有 CPU 源码和边界审查，未运行 GPU、服务、Math 或官方 benchmark，未采用。
不修改正在评测的 K8V16 profile。先等它完成 Math 和正式4K，再完成独立微测。

旧 BF16 地址32实验只覆盖 B4..48，正式4K没有明确收益；不能据此推断 K8V16/B64 的表现。
新候选保持相同 K8V16 12KiB 物理页、编码、writer、浮点计算、线程配置和64位物理指针，
仅把已测 reader 中28个 stride 注解和6个元素偏移 cast 从64改32。
源码SHA锁定并以 AST 确认只有34个整数类型节点变化。

任何 GPU 入口必须先对 Q/K/V、输出、页表、各 metadata、softmax scratch 和 scale
检查真实 shape/stride 的最大元素偏移，含保守 masked lane 余量，小于2^31才可选候选；
超界保留原64位路径。公开 slot mapping、索引API和物理指针保持原类型。
需要相同页表和输入的当前K8V16/地址32两组，在变数据同Graph、完整物理存储与RNG检查、
自身FP32和CPU参考之外，还要求两组输出逐位相同。未完成这些证明前不进入真实模型。

两组8场景微测与独立离线审核工具已准备，真实 Producer AST 的36种 CPU 分支通过。
尚未 GPU 编译或运行；先执行已准备的11参数 K8V16 sweep，再判断是否测此候选。
