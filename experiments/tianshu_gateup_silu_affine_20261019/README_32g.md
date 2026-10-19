# BI-V150 32G：独立affine GateUp/SiLU验证准备

只读复制旧未提交原型中已修正constexpr的affine核到新目录，不修改或提交旧两份WIP。两路连续weight loads、两个FP32累加器，投影先舍入BF16再FP32 SiLU/product，最后BF16。计划B64模型宽6144、两种尺度，B17真实宽和130尾部，共4场景×7配置；仅数值通过且改变输入的同Graph、独立CPU点和padding通过才计时。记录实际编译warp/stage/寄存器/spill。当前服务没有融合绑定，GPU尚未验证，无Math/吞吐收益结论。
