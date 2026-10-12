# BI-V150 32G：2步概率提案标准小规模负收益

保留7槽训练骨干，LM head/Markov只取前2步，target验证2proposal+bonus，正FP32阈值、标准概率拒绝、target Graph/draft eager。修复物理lookahead至少7及positions/slot_mapping/arange容量；C8实际smoke与8个attention场景/81CPU检查此前通过。

4K/1K、C64/N128三轮total tok/s：1430.960098、1938.891673、1948.913980，384请求全部成功，usage逐项核对4096输入、1024输出。排首轮均值1943.902827，对同规格最佳控制2731.418083变化−28.831736%。比真实7步的1660.326231有所改善，仍没有对当前最佳的收益。

调优104/22/24，后两轮仍现场调优，因此均值是当前实际服务效果，不称纯稳态差距。没有采用、独立Math或正式4K；不能沿用旧FP64错误接受的5254/8133成绩。

完整14413352 bytes、55分块及整包SHA256 c9b84c2f0e16bc9c3da74e91eb7e0c3b477bccf8d608e145aea146cf1a8dfea6本地核验，保留384响应、每轮指标、全日志与冻结源码。正式NoSpec最好仍2712.55、TTFT4720.38ms、Math102/105。

下一步尝试vLLM原生draft greedy，主模型仍temperature1/top_p.95且标准随机拒绝，减少draft完整概率缓存与随机采样开销。新候选尚无性能/质量结论。

性能剖析：TODO。未修改训练权重、安装源码、原始benchmark和沐曦仓库；未运行优化版完整16K。
