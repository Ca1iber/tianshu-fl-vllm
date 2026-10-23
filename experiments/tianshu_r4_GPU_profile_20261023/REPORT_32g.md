# r4 CUDA 时间线诊断

16帧 FULL/C64 纯decode、两波4K/256、128请求全部正常完成，实际r4的944个来源未改；源、快照与trace SHA由独立CPU重核。观察320个GPU kernel events及16次cudaGraphLaunch。此后端未展开模型Graph子kernel，因此不能由此宣称attention/GEMM占整个模型的百分比。

图外中位数：TopP GEO32K/16K约2445.7us，W4 LM约521.1us，FP32 true_div rank2约350.3us，rank1约191.2us，softmax179.8us，exponential100.2us，argmax87.0us。仅诊断时间，不能当benchmark得分，GPU事件和CPU耗时也不直接相加作为关键路径。

同seed777、同128token输入：先冷缓存与后新salt冷缓存输出全文相同；复用salt后末两token不同。实际命中计数复用增加112token，新salt增加0，查询均增加128。证实该文本差异与本次冷/热缓存条件相伴，不把潜在混合精度/shape路径作为已确认根因，也不把此前CPU profile失败记录改为通过。模型原r4发布与Math结果保持原证据；本诊断不是新的模型资格或吞吐成绩。

完整trace/result/指标在容器 `/workspace/benchmark-results/20261023_r4_GPU_profile_tianshu`，Git仅代码、报告和关键JSON。服务正常退出、GPU恢复空闲。
