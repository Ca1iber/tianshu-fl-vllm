# RMS与LM组合C64：101/105，独立全量审计通过

新鲜Math-500 Level3 C64正确101/105=96.190476%，严格>95%；105请求全部成功，全部stop、无max_tokens封顶。错误索引18、23、87、98。该组合只跑此首次C64，未同版重跑挑分。

原始运行phase complete、evalscope exit0，独立CPU从全部105条review/prediction/report重新计分。944模型输入、91控制器及首次GPU前快照、原checkpoint SHA、原13serve参数与63环境均核对；全部继承attention/KV/scheduler/sampling/GU/LM/Down/QKV/O及新RMS+A8、新LM运行审计重新执行，完整profile和accuracy_audit与原始完全相同。42层实际norm/原GU图捕获及CUBIN、LM实际M64新二进制/原pack/Profiler峰值通过，未冒称旧独立A8或旧LM64执行。C64实际QKV/O及Down M4096各42层。

这是部分质量结果，QUALITY_GATE.model_quality_qualified=false，不能放行benchmark或采用。C8已经启动；后续必须两批各105严格>95%和完整独立CPU双批次审计通过，才运行新同期短4K。原始记录只容器，GitHub保存代码、105条关键得分、门槛和本报告。
