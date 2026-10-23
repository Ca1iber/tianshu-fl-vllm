# r3 + M4096 QKV RMSNorm/A8融合的新鲜质量验证

唯一新路径是42层静态M4096 QKV前的norm+A8融合。原r3的903项模型来源全部逐字保留，原13项serve参数及63项环境契约继续核对；compile_sizes实际新增4096，旧安装阶段的compile_sizes声明不代表新全局配置。未合入RMS64/LMpad组合。

先C64再C8，各一次新鲜Level3全部105题；严格>95%，所有错误及截断计入，不重复抽分。C64沿用既有FutureKV control质量覆盖政策，C8与发布策略相同。全量旧KV/调度/采样/Down/O/GU/LM/QKV其它M守卫和新42层融合、原pack指针、真实norm/dequant CUBIN、实际native memory profile显式探测、通用与M64图不变均检查。兼容记录仅表示真实相同MM+epilogue，明确未调用旧独立A8或原custom_op。

首次GPU前冻结模型、控制器、独立CPU复核器；复核全部105条原始review/prediction/report，重新计算完整profile和accuracy_audit并逐字段相等。新鲜双组均合格后方可短4K；此入口不运行正式吞吐或16K。截止2026-10-23 15:58 UTC，原始文件只在容器，代码/报告/关键结果GitHub。
