# 严格TopP预筛选：CPU支持集验证通过，GPU准备完毕未运行

父实现就是当前正式版本的几何中点TopP（09156e66...），相同32K/16K tile、w16/s1、32轮搜索上限、1e-12区间容差、1e-6独立CPU语义门槛。只改standalone TopP的初始outlier候选阈值：原avg+std*sigma，两个候选分别额外加0.5*std²或std²；TopK分支AST、完整softmax、sum_outlier_probs>p判定、候选不足时全词表回退、后续阈值搜索、重复值处理和最终mask原样保留，p和RNG不改。

动机来自exp(x)乘正态密度后的加权均值mu+sigma²，只用于提出更紧的初始候选，正确性不要求logits服从正态。只要候选集合的真实概率质量>p，所有精确排序nucleus token都在这个高概率集合内；否则沿用已有全词表路径。CPU独立排序检查覆盖4种长度、10种normal/wide/peaked/bimodal/ties/flat/masked/one-finite/all-inf/heavy-tail输入、5种p、3组，共600次支持集保留；这不是Triton浮点、速度或模型质量证明。

CPU_GATE_32g.json保存完整7份依赖与生成源码哈希、600检查计数和原始完整案例文件SHA。CPU600案例及生成代码/快照留容器20261020_topp_preselection_CPU_v2_tianshu，首次CPU原件也保留；GitHub保存精简关键结果。

GPU probe已准备12场景×3组，包括B1/8/32/64、正常/宽分布、mask、tie、BF16来源logits、全相等、TopK100+TopP，以及逐行p变化。每组都要求独立CPU FP64最小nucleus语义：mass>=p-1e-6、去掉最小保留概率后mass<=p+1e-6、无更大logit被移除、原有限值不改；同一Graph改变logits与p并检查与直接执行一致、RNG状态不变。7次轮换×10 replay包含相同输入恢复。源码/快照/编译器与实际kernel元数据会冻结。

run_probe.sh是直接shell入口，要求真实GPU空闲，结束后保留退出码及硬件状态。当前只准备好，没有排队、没有运行GPU、没有服务/新Math/正式结果、未采用；须等当前小暂存区→地址32→内存组合服务全部释放GPU，或根线程明确调整等待队列优先级，再考虑执行。任何组件收益仍需4K服务筛选与最终配置全新Math严格>95%。
