# 当前 GEO TopP 在线归一化：严格数值通过，无明显组件收益

2026-10-21，BI-V15032G。新对照是推荐版实际使用的 GEO 源 SHA09156e66...，32K/16K、w16/s1、原min(B,16)网格。候选 SHA c4a2dd8d... 仅把 standalone TopP 最大值扫描与随后真实最大值的指数归一化扫描，合并为稳定在线 max/sum 递推。统计预筛选、TopK AST、几何搜索32步、1e-12区间、全词表回退、p/温度/RNG、最终mask未改。FP32归约顺序改变，不声称逐位完全相同。

120组CPU FP64数学/源反转检查通过；GPU12场景×2方案×2输入阶段通过。独立CPU重新读取全部完整输出，重算1444行、188528640个FP32值对应的FP64softmax、排序、质量、最小支持集和原有限值保留；严格容差仍1e-6。两方案均通过，少数场景掩码有逐位差异并保留原记录。同一图改变logits/P后与直接调用一致，RNG不变。组件B1仅边界诊断，不表示实际推荐版会把B1送到GEO。

|场景|当前GEO ms|在线归一化 ms|候选耗时减少|
|---|---:|---:|---:|
|B64 FP32普通|2.595360|2.603314|-0.3065%|
|B64 FP32宽分布|1.678054|1.674120|0.2344%|
|B64 BF16普通|2.469568|2.461367|0.3321%|
|B64 BF16宽分布|1.896671|1.895743|0.0489%|
|B8动态P|0.619759|0.617861|0.3063%|

七次轮换、每次10次热Graph replay，每次含相同完整输入恢复。主要场景变化在约±0.34%内，没有值得进入模型筛选的稳定收益证据；不跑本候选服务/Math/N256，不采用，不改当前5586.28推荐成绩。这不是完整Sampler或模型吞吐，不把节约一次扫描等价为运行时间必然下降。

## 输入落盘错误与CPU复核恢复

首次冻结的probe.py把原阶段input文件在改变logits/P后又保存了一遍，导致 original.pt 与 changed.pt 都是第二阶段输入。GPU实际内存计算/数值诊断、两阶段完整输出和计时未受该存盘命名错误影响，但原离线验证器因此失败；初次失败日志保留在独立CPU恢复目录。冻结的GPU源码、CPU_GATE和原件均未修改，GPU未重跑。

独立 `tianshu_topp_online_sum_cpu_recovery_20261021/verify_gpu.py` 核验冻结源确实只有一次词表维flip与动态P的batch维flip，并检查两个输入原件完全相等；利用flip的精确自逆性恢复第一阶段CPU输入，再逐个完整输出重算全部诊断。恢复结果明确记录 original_input_record_was_overwritten_by_changed=true、original_input_reconstructed_by_frozen_involutive_flip=true、no_GPU_rerun=true，不能冒充原记录器无错误。

GPU原件：`/workspace/benchmark-results/20261021_topp_online_sum_GPU_tianshu`；完整CPU结果：`/workspace/benchmark-results/20261021_topp_online_sum_CPU_recovery_tianshu`。完整张量、编译产物和日志留容器；GitHub只代码、报告和关键JSON/CSV。原实验GPU启动后源码冻结，用户两处dirty未改。
