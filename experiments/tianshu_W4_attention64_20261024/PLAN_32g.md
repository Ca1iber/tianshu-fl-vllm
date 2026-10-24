# 主M64 QKV/O：对称与非对称G128 W4

r4正式Graph64的QKV/O均BF16（实际QKV2560×2048、O2048×2048），占历史FULL decode回放98.58%。此组件仅实际BF16 vendor mm 对照、native dotsaddz W4 symmetric RTN与affine minmax三路。G128编码在真实BF16 scale/offset上取uint4，并按原生实际BF16_round(q*S+Z)验证；不做训练、微调、checkpoint改写或根据Math答案校准。G64已有明显负性能结果，本轮不重复。

各role均42个不同真实bank/独立X计时，三phase合成BF16激活，L0/20/41完整FP64原始与量化参考，其余bank完整输出/重复/Graph/input检查。旧84-bank BF16 SHA及r4真实M64捕获42QKV+42O记录绑定后才运行。数值阈值同原W4组件：逐格 .012abs(ref)+.006RMS，relL2<=.006；完整输出而非抽点。全部量化格式pack/unpack逐元素CPU核验。原生CPU/GPU RNG不变、权重/输入全bytes/versions/guards检查、同一Graph变化输入，poison后重放必须与eager逐位。先保留输出及任何失败，不原样重跑追分。

QKV/O分别测eager/Graph各7轮交替三路，每轮10次42-bank调用，预设排首轮，全部时间入结果；计时不并行CPU准备或归档。计算全部3层3phase投影误差，对称/仿射不借用静态权重误差当正确率。组件至少5%合格Graph减时且热态spread<1%才考虑主批次模型筛选；若无收益停止该角色。模型接入还要原生load/profile统计全部副本和scratch，并保持FULL/64/uniform/noLoRA范围以免偷偷改M64 prefill。无新模型质量/吞吐，官方N256必须新Math>95%和正向短4K后；本轮不跑16K。

原始数据在容器SHM，完成后归档持久目录，Git只代码、报告和关键结果。r4原944来源不改，官方13argv保留。GPU最多1800秒，截止2026-10-24 23:58。
