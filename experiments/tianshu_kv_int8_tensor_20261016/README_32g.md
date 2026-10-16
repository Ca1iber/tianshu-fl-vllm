# 天数逐层静态INT8 KV组件探索

此前逐token/head INT8含循环内scale读取或反量化开销，未获加速。本版手工INT8固定power-of-two逐层K/V scale，比较原BF16、沿原生per-tensor反量化语义的INT8和scale外提：K scale并入softmax_scale，V scale在分段归约后只乘一次。存储是INT8，per-tensor参考虽借用KVQuantMode.FP8_PER_TENSOR分支，但不是FP8数据，也不是框架已支持的INT8 dtype集成。

保留当前.cv/page16/B64S8，独立GPU微测3场景与两轮改变Q/K/V后清空输出Graph检查，独立量化FP32/CPU参考、相对BF16误差、clipping统计与7组事件。计时不含量化writer和模型服务。仅有明确组件收益才进一步设计真实框架allocator/writer/metadata路由及逐层校准，之后必须新Math105>95才能采用。最佳4093.81/Math102不适用于量化候选。

不改权重、安装源码、最佳启动器、原benchmark或沐曦仓库；无16K，2026-10-16 23:58硬停止。
