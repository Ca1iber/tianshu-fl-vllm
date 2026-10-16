# 天数纯Decode组批元数据融合

从用户指定的KV cache、batching、scheduler方向入手。当前正式无KV抢占，缓存提交已有“无新完整块”早退；新的候选优化ModelRunner的batch元数据准备。仅无投机、无多模态、async开启、纯1token Decode、前后请求位置一致且已经完成prompt时使用。KV分配、页表和slot mapping仍调用原实现，CPU bookkeeping保持。

一个小kernel准备当前位置、seq_lens、query_start_loc、accepted计数、req_indices、query_pos、scheduled计数、discard mask、上一轮采样的input_ids和logits_indices；positions及req_indices仍为int64。仅省去独立GPU操作与重复H2D，不改页长、缓存dtype、调度预算、maxseq或采样算法。不支持的batch继续原完整方法。

微测20组、6个guard、2次动态输入Graph通过，完整准备host中位数2.023→0.321 ms。slot mapping使用stub，尚无模型吞吐结论。结果见results/20261016/micro_32g；已准备真实服务3+3对照。

真实服务3+3已完成，4035.832→4039.083 tok/s，仅+0.0805%，不采用。详见results/20261016/service_32g。
