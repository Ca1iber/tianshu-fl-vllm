# 天数 BI-V150 32G：K8V16 KV load cache hints

Local runnable component experiment, not an official max-shape score.

只改当前 K8V16 reader 的 KV_CACHE_MODIFIER：current=.cv、cg=.cg、ca=.ca、default=空。物理页12KiB、K编码、V BF16、writer、launch与分段策略保持一致。旧 BF16 页的缓存结论不直接外推到混合12KiB页。

GPU计划：原9形状，包含1/8/16/24/32/64批次、4K/5K上下文、短33、跨stride页。每形状相同输入及随机页表，原writer+decode同图，修改Q/K/V/slot/长度后重放；各臂完整输出与current逐位一致，否则拒绝。保留CPU FP32抽样向量、完整GPU参考误差、存储/输出/scratch guards与RNG检查。实际读取已编译ASTSource中的KV_CACHE_MODIFIER，记录compiled kernel hash，7组旋转次序事件中位数。

CPU检查执行实际Producer AST，72分支，所有launch参数相同，仅constexpr缓存提示不同。未跑GPU前不声明收益。该组件不安装完整引擎，不声称Math或模型正确率。GPU profile尚未采集。只在稳定收益明显时继续服务对照与新鲜Math，当前正式最优4997.05 tok/s与公开2028.01 tok/s均不因此更新。
