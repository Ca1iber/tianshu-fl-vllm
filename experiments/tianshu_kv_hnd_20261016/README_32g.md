# 天数KV布局NHD/HND组件实验

新候选保持逻辑[num_blocks,2,16,2,128]、BF16、page16与原writer/attention，只把物理[token,head,dim]改为[head,token,dim]。实际Triton backend原生提供NHD/HND stride order，writer传入page_stride/head_stride；此前仅测过16/32页及K/V planes，尚无HND结果。

基线使用当前GEO最佳的完整38项组合，attention为.cv/L0/KV16-w1/S8；先控制GPU微测，16个真实builder/writer/forward场景，覆盖prefill/chunk/ragged/mixed/decode、边界与非连续物理页。两轮原地改变Q/K/V、slots和长度后Graph重放，对FP32/CPU参考、整个物理KV存储、输出和segment尾部核验，7组AB/BA事件时间。源码及全部依赖快照保留。微测仅决定是否进入服务A/B，不直接作为模型收益或准确率。

尚未运行。必须等待Scheduler/KV诊断终态、服务关闭和GPU空闲后串行执行，服从23:58硬停止。无新16K测试。

16场景已完成并本地复核61源码、224事件：B64/4K约-0.0022%、5K-0.0638%，基本持平，不采用。报告见results/20261016/micro_32g。
