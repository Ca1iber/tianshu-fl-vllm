# dotsaddz 非恒定 GPU 格式找到唯一精确匹配

30个事前冻结假设全部正常结束，唯一完整16384格输出精确匹配为 `n_natural__unsigned_add`。对应 W int32 [K,N/8]，沿N自然顺序打包unsigned uint4 q，低N在低nibble；S BF16 [K/128,N]；Z BF16 [K/128,N] 为 -logical_zero*S。实际数学语义为 q*S+Z。调用 `aten._weight_int4pack_mm_dotsaddz(x,W,128,S,Z,size_n=N)`。

其余29个假设均为数值不匹配，没有异常或超时。所有操作均保持输入完整字节、版本、初始化保护空间不变；参考是同一非恒定矩阵、两组不同scale/zero、全K INT64投影，随后独立CPU复算全部输出。原始过程：`/workspace/benchmark-results/20261023_native_dotsaddz_formats_{CPU,GPU,runner}_tianshu`，全部30进程状态均exit0。

这只证明该实际安装库的格式正确性，不证明速度、真实模型权重误差或Math500质量。源码冻结b832f2496969ce0b696a421b2ff0fabc0cb016f1；没有改动原始checkpoint/服务，沒有正式benchmark。下一步复用相同逻辑量化权重的42层Down64组件检查绝对耗时。
