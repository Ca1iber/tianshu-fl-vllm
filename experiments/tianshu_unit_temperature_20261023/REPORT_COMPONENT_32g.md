# 中性温度：实际Sampler前置检查通过

精确T=1从实际InputBatch CPU请求元数据判定，unknown metadata、非全随机、非单位温度或tensor身份不同回退。ContextVar限定到当次Sampler.sample；不读回GPU、不改变温度/p/RNG/模型。8组包含B1/B8/B64、1/.7/1.4/混合、default/部分/全部逐请求generator。完整温度缩放bits、最终token、global/各generator状态和BF16 logits输入均与同次原版完全一致。已CPU核对所有记录来源SHA，见COMPONENT_CPU_GATE。

首probe在GPU测试输入创建前因继承钩子的sys.path把`import policy`指向其他实验同名模块而失败。probe_v2按本目录绝对路径和唯一名称加载；策略、阈值、用例不改。两份源码与失败原件均保留。此处仅前置组件验证，不是新Math或正式性能资格；独立服务对照在screen实验。当前正式发布仍r4。
