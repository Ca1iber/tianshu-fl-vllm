# 天数32g：准备Decode寄存器上限与驱动计数诊断

实际安装的triton/backends/iluvatar/compiler.py声明maxnreg，并把它传为iluvatar-num-vgpr函数属性；通用CompiledKernel从驱动load_binary读取n_regs/n_spills。不能用NVIDIA通用文档代替该实际后端证据。探针启动时核验这些源码特征并冻结两份源快照。

保持当前KV16/w1、B24回退、B64S8、BM16/BQ2、stage1和相同grid，比较默认/64/96/128上限。实际记录requested_maxnreg、compiled_maxnreg、n_regs/n_spills及shared；选项被接受不等于有效降低寄存器，不根据这些字段直接声称占用率提升。

CPU实际Router与生产LaunchRouter执行764组，逻辑query网格/reducer和上限参数传播通过。GPU计划17场景×3候选，FP32/CPU及Graph改变query检查、尾部、全事件计时；未运行。只有上游预算对照成功归档并终止且GPU空闲才启动；预算若>=2%则held优先独立确认。
