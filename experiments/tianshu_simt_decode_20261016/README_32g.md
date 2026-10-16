# 天数：SIMT GQA decode核心筛选

独立候选从当前cv8deb生成，普通3D/GQA8/h128/BM8/BQ1路径把QK与PV的tl.dot改为显式BF16乘积的FP32求和（P仍先转BF16再乘V），并使用每序列一个query CTA的直接映射。保留原在线softmax/.cv/L0、int64、KV16、S8和原reducer，不用log2候选。浮点归约顺序、producer矩阵行布局与grid均改变，不能宣称逐位一致或单项机制收益。

核心只先测4个场景：B64/4K、B64/5K-strided、B64/65-padding、B8/33。对照当前BM16 Tensor路径，另保留旧BM8 Tensor/warp1参照；新SIMT扫描warp1/2/4/8，合计20候选场景。每项独立FP32/CPU、原/变QK Graph与尾部检查，9轮交替event、实际metadata/编译产物。若核心已慢则不扩大边界或进入服务。GPU尚未运行，不能将CPU1952映射/可逆源码/CLI检查当数值通过。

运行：`python3 experiments/tianshu_simt_decode_20261016/run_probe.py --output NEWROOT --stop-at-utc ISO`。串行GPU、冻结10源和明确截止。即使核心通过，仍需完整边界与新Math>95%才能采用；当前正式3992.97、Math103保持。
