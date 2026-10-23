# 原生INT4接口CPU核验：可测试，尚无GPU结论

实际Torch2.10调度表显示通用INT4打包/矩阵乘以及CoreX额外AWQ、dotsaddz CUDA入口已注册，with_scales_and_zeros入口没有CUDA直接实现。查询全过程CUDA_initialized=false，安装_meta_registrations.py源码SHA固定。

只对通用入口准备4个64×256×256确定性功能例，CPU Meta确认uint8[N,K/2]→int32[32,2,32,4]→BF16[64,256]，group128、inner_k_tiles8，独立预声明65536个输出期望。GPU首次执行前冻结；排队脚本仅在prefill质量/短测序列PID41925终止且GPU空闲后运行，90秒上限。不改变模型，不声称权重质量、实际硬件支持或性能提升。
