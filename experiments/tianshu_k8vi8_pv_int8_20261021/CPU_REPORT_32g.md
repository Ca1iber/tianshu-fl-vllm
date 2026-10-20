# 整数PV原型：CPU参考与源变换通过，尚无GPU结论

此原型沿用已有K8/V8真实writer及8448-byte物理page，在PV阶段先把V的逐token/head scale并入P，再按每query/tile行做FP32 RNE量化与INT8×INT8→INT32点积，恢复FP32并沿用原online-softmax累积。QK计算、softmax/reducer、公共API与缓存写入/布局不变；默认flag=False仍可作为原V8对照。

源变换的每个修改可逆恢复到精确父文本；API/reducer/cast函数AST检查通过。CPU使用独立IEEE FP32 pack/unpack与ties-to-even计算，核对7168个量化值；独立Python整型累加核对12288个INT64输出，6组T32/64/128、zero/tiny/changed-scale案例均满足推导的误差界。T128的最坏整数绝对上界2064512，小于2^31。CPU全局RNG保持。

这些检查验证源变换和CPU oracle，不证明Triton在GPU上的正确性、吞吐或模型质量。尚未GPU编译/计时，也没有接入engine、跑新Math或采用。后续先对实际GPU primitive的完整Pq/scale/INT32/output做诊断，再测完整真实writer+attention与改变输入的图重放。数值参考使用CPU FP64，不采用此前已发现异常的GPU FP64量化路径。

只有组件有明确收益才考虑真实KV生命周期/allocator/prefill/模型接入与fresh Math>95%。本次准备完全在CPU，未打断预算8192的Math→正式4K串行队列。
