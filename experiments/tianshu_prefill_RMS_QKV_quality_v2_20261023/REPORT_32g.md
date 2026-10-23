# v2启动失败：新增静态范围没有继承正式native GU图替换

冻结60a137b，13:29启动，13:34在原native GU的capture守卫停止，尚未发出Math请求。新QKV融合42层真实profile和norm/dequant CUBIN已命中，显式无残差/有残差scratch probe位于真实内存窗口，RNG/权重不变；但旧RangePass/NativePrefillPass仅处理[1,4096]，新增[4096,4096]中GateUp仍为linear+SiLU，原GU profile native_calls_after=0，42层守卫正确拒绝。不能宣称模型正确率通过。

原始结果`/workspace/benchmark-results/20261023_prefill_RMS_QKV_v2_math64_tianshu`保留，旧源码/守卫不修改；排队短测因无独立质量Gate正确跳过。独立v3在新静态范围复用原匹配器直接恢复同一native GU custom op，原general/M64范围继续原pass；原86份pass记录和新范围43份记录分别真实审计，不伪造旧调用，不放松旧GU profile守卫。正式最好仍r3 5950.75 tok/s。
