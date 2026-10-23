# RMS+A8 GateUp v2：完整双批次质量合格

| 新鲜Math-500 Level3 | 正确/总题数 | 正确率 | stop/max_tokens |
|---|---:|---:|---:|
| C8 | 103/105 | 98.095238% | 104/1 |
| C64 | 101/105 | 96.190476% | 104/1 |

两组均严格>95%，105请求全部成功，原始evalscope退出码0。两组索引101均达到输出上限，全部照常计入；C8错误18、87，C64错误18、87、99、101。遵从原始逐条官方评审，不把cap自动删去或改分，不同版本只各跑首次新鲜质量，未挑选同版重跑。

916项模型输入以r3原903项为精确字节子集，80项质量控制器在GPU前冻结。只改变静态M64 GateUp前42层IR-native RMSNorm+A8融合；QKV64/O64/Down64、QKV128/512/4096、O512/4096、LM和其余框架沿用r3，原checkpoint/结构、13serve参数和63环境保留，C64既有FutureKV control质量政策未改。

独立CPU对全部原始review/prediction/report重新计分，核验所有模型/控制器及GPU前快照和checkpoint SHA，并重新执行全部继承attention/KV/scheduler/sampling/GU/LM/Down/QKV/O与新RMS+A8运行审计，完整profile和accuracy_audit均与原控制器结果完全一致。真实42层norm与原GU GEMM捕获及CUBIN一致、图重放关联、旧独立A8未伪造、动态范围原字节均通过。C64实际QKV/O和Down M4096各42层；C8只命中QKV M128，不将未击中的M4096路径冒称已验证。

QUALITY_GATE.complete及model_quality_qualified均true，C8/C64各105的新质量已合格；仅使此版本可以进入同期短4K对照。当前N64三轮排除第一轮的r3控制组正在运行，尚无候选吞吐结论。RMS组件带残差25.69%延迟降幅不当成模型吞吐提升，没有新正式成绩或发布，最佳合格仍r3正式5950.75 total tok/s。

原始输入/响应/日志/数组仅天数容器/workspace/benchmark-results/20261023_RMS_A8_GU_v2_math{8,64}_tianshu。GitHub只存代码、逐题关键分数、门槛和报告。
