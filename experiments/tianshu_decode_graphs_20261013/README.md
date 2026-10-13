# 天数32g Decode图范围实验

已实际验证仅将Decode graph上限变64、11点，独立Prefill75点warmup及运行padding不变、FULL_DECODE_ONLY保留。真实N64/C64/4K1K五轮排首两轮平均3235.265084 tok/s，对刚控制3253.050661为-0.546735%，KV只增加144tokens，7greedy相同。无清晰收益，不采用，不重复正式评测。

结果和完整SHA归档：results/20261013/service_32g/REPORT_32g.md。原型是可回退的进程hook，未修改安装源；所有新候选均需独立质量与正式分数才可采用。
