# K8 word读取：先解码，再组合BF16矩阵

这是独立组件候选。当前已验证pair_rows为控制，比较四字节读取后的BF16先解码再join（行/列两方向）和pair_rows的warp2。此前四字节负收益实现先把uint8字节重组成完整矩阵再解码；本次改变的是中间dtype与组合顺序，不能沿用旧结论或预设收益。保持12KiB物理页、K8/V BF16值与writer、S16策略、tile16、.cv和reducer；warp2是单独对照，其他模式warp1。

CPU逐lane穷举全部256码值及源码可逆/ABI验证；GPU延用完整12场景、四路、两相位变数据同Graph、FP32/CPU参考、逐位输出、整个缓存和边界/RNG检查，432组交替event。编译IR/PTX只留容器。服务对照正在占GPU，本组件尚未启动；只能在确认其实际进程终止、结果成功、GPU空闲后运行。
