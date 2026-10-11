# 天数框架metadata整数融合（服务A/B中）

只融合实际ModelRunner的三处写入：positions=index(counts).to(int64)+query_pos，seq_lens=count+scheduled，seq_lens尾部清零。输出dtype与原实现相同，无模型权重/注意力/采样改变。涉及int32溢出、转换后int64位置加法、256/1024缓存和13项尾部保护，15/15 CPU bitwise/FlagGems参考检查通过；含真实1..8192 token长度和1..256请求数。

GPU Graph微测从约12–15us降至约2.2us，不当作完整服务收益。进程级hook仅在显式旗标1时启用，要求实际ModelRunner SHA b0144bcff96bef15b04eb1ad45385b01c47faea056bc56352a6486b7c497fa2e，AST精确匹配一个三语句块；其余函数内容保留。仅适配Iluvatar BF16 MiniCPM42层/2048/GQA16:2，TP/PP/DP/DCP1，无Quant/LoRA/spec/MM/MRoPE/XDRoPE；不支持时保留原有三处写入。尚未写site-packages。

服务对照保持tile128/adaptive1/budget2048，只有metadata0/1改变；4K/128、C64/N192、三轮排首次。需CPU验证源一致、实际metadata路由记录、原始ModelRunner与tile源一致。正在先跑控制组，整体性能和新配置对应官方数学尚未完成。当前正式最佳检查点仍tianshu-best-fa2tile128-20261012。所有GPU任务串行，11:58停止并备份。
