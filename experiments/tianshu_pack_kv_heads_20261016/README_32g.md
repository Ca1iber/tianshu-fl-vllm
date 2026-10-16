# 天数：同序列两个KV head合并的Decode矩阵块

待GPU。BM16的16个query行分别映射到2组各8个query heads；每步逻辑16个KV token打包成两个head的32列。softmax前屏蔽跨head分数。序列轴沿用native解析，head轴2→1；BF16 dot/FP32 acc、L0/.cv、int64偏移、KV16/S8、原epilogue/reducer函数体保留。原实现与False恒等对照，True比较warp1/2/4，先筛4个核心场景，不作完整边界或模型成绩声明。
