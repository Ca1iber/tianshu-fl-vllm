# 天数Scheduler/KV只读调用诊断

当前最佳GEO+M2048原serve参数，采集两轮4K/1K C64 N64。完全调用原Scheduler.schedule及KV方法，不改变队列、预算、KV页长或缓存策略。每step记录原调度wall时间、排队/运行请求、剩余KV块、prefill/decode token拆分、allocate_slots空分配比例、SingleType cache早退比例。嵌套方法计时有重叠，包含observer开销，不能把它们相加或视为无插桩吞吐。

14份实际安装框架源码SHA来自只读审查清单。每128步刷盘，最后flush marker+sentinel保证结束前记录落盘。尚未运行，待当前batch metadata对照完成后串行执行。

V2仅修复V1包装函数漏转发参数；check_cpu.py核验安装签名并验证默认/位置/关键字三种调用与返回identity。V1失败日志保留；V2待运行，不声称已有诊断结果。

V2已完成2349步及128响应，本地复核75源码/39环境通过；结果见results/20261016/trace_32g。无KV分配失败，256个等待中的吞吐调度步均用满2048token预算；不直接据调用计时改调度策略。
