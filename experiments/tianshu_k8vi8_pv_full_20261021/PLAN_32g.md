# 整数PV完整attention：预先固定算法、参考和测试范围

当前407源推荐版/62环境键/原13项serve为配置和源身份父项；实际活跃reader f1df731b、B64/4K为S4/tile16/w1，已从正式结果核对。本轮只做单层组件，不安装完整engine/cache allocator，不声称Math或模型吞吐。

复用已经独立GPU/CPU通过的整数PV原语：source.py、kernel_pv.py与旧原语逐字节相同；V8真实writer/RNE参考与已验证版相同。保持QK/softmax/归并计算，PV阶段将P乘每token/head Vscale，逐query/tile absmax/RNE量化P到0..127，INT8×INT8→INT32 dot后恢复FP32累加。此算法引入额外注意力近似，最终采用仍需新的Math105>95%。

七条路径：未改动的真实当前K8V16 reader、原生V8 BF16消费reader、新reader的PVFalse、PV32/w2、PV64/w2、PV32/w4、PV64/w4。PVFalse必须全输出逐位等于原生V8；B128的2D回退四候选也必须逐位等于原生V8。候选只用于非TD、非滑窗、3D单query。

沿用10个声明点：1/33、1/4096、8/4096、16/4096、32/4096、40/4096、40/16384、64/4096、64/5120、128/4096。40/16K是单层组件，不是完整16K benchmark。每点三个阶段改变Q/K/V、scale、slot和长度；相同随机物理页表、contiguous/strided真实FullAttentionSpec布局、负slot与未用页/输出/scratch/RNG保护。7轮七路旋转、每轮20次热图重放，包含真实writer、量化和scale写入，不计allocator/源编译。全部10点/210路径要完整保留，任何编译/实现错误使原始任务失败；不得事后删除失败点。

数值参考在GPU执行前固定：完整输出对独立Torch分段FP32/RNE/整数PV模拟，实际整数乘积和在tile×127²<2^24范围内可用FP32精确累加。选中CPU向量另外从保存的原始BF16 Q/K/V用NumPy FP64 score、显式FP32 exp/RNE和INT64 dot重算。整数算法比较为atol=.001、rtol=.005并且relative-L2<=1%；同时检查量化误差解析上界，加BF16最终舍入与5e-4浮点留量。独立CPU重新读取真实GPU输出，不只信任在线expected或passed字段。

原版/原生V8维持既有理想attention门槛atol=.02/rtol=.01。新整数近似相对理想attention的同门槛误差全部另记（含失败点），不将它冒充逐位目标，也不据组件算法门通过声称模型质量。原始BF16注意力精度差异另记。只有实现/完整KV写读检查通过、组件收益足够时才考虑engine接入和独立Math>95%；不重跑失败Math刷分。

每阶段保存选中序列完整原始Q/K/V和七条路径实际输出，CPU可重建2184向量（其中1104整数PV向量）；writer额外保存真实K字节、V整数和scale。失败时尽力保存该点完整输入/输出，原始PT/日志只留容器，Git只存代码/报告/关键JSON/CSV。硬停2026-10-21 23:58北京，不改旧冻结实验、安装源、推荐版或沐曦仓库。

CPU预检分两部分记录：独立纯CPU子进程运行两套数学参考并确认CUDA始终未初始化；原生FullAttentionSpec布局/框架身份检查会因平台插件导入初始化设备上下文，此事实单独保留。后者只使用CPU存储做布局核对，不执行本候选GPU kernel。首次预检失败只在“导入框架后CUDA未初始化”的元数据断言，192个参考向量与布局已通过，日志保留；未把设备上下文初始化伪称为GPU数值验证。
