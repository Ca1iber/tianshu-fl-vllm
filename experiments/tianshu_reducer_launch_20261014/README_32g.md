# 天数32g：Attention分段合并独立warp配置

在已采用.cv/S8/guarded KV16/w1 producer基础上，唯一候选变化是原生reduce_segments launch的num_warps，比较默认和1/2/4/8（含显式4控制）。原生reducer源码、producer代码/配置、grid、tile、segment及地址宽度不改。归档backend默认warp4/stage3是静态依据，本轮仍会记录真实编译metadata，不以静态默认冒充运行值。

CPU检查执行真实probe adapter和原有LaunchRouter/_ReductionRouter，使用假的native/JIT确认producer始终KV16/w1/stage1/.cv、reducer tile同步16、仅reducer注入warp参数，五路由通过。fixture不是GPU证据。GPU计划17场景（4K/5K、strided、query填充和tile边界），包含FP32/CPU、原输入/改变QKV后的Graph、尾部、9轮交替全Attention设备event及B64 producer/reducer实际编译产物。warp布局可能改变浮点归约顺序，不能预先宣称逐位相同。

run_after_grid.py确认原CTA排列控制器实际退出、complete且归档存在、GPU空闲后串行启动，冻结六份运行源并受同一租期截止约束。没有新吞吐/Math/正式分数，不改变当前4091.29最佳配置。
