# 天数组批元数据融合服务对照：仅+0.08%，不采用

微测20组、6guard及2次动态输入Graph通过，完整准备host中位数2.023→0.321ms（KV slot mapping为stub）。随后真实模型保留原KV分配、页表、slot mapping、CPU bookkeeping与int64位置/请求索引，按原serve命令运行3+3轮4K/1K C64 N64对照。两边均为当前GEO+M2048最佳算子、75warm、budget2048、maxseq256及async。

|模式|第1轮（排除）|第2轮|第3轮|后两轮均值tok/s|
|---|---:|---:|---:|---:|
|control|3841.326653|4036.948934|4034.715098|4035.832016|
|candidate|3850.698463|4040.620267|4037.545004|4039.082635|

稳态增益+0.080544%，没有明确服务收益，不采用，不启动新Math/正式benchmark。不能从本次结果定量断言CPU/GPU重叠是原因；本次只证明微测下降没有转化为明显吞吐增益。

67份源码快照与当前自有文件SHA、实际40环境、唯一开关差异、生成方法SHA/可逆替换、全部384响应正文/usage/seed/TTFT/耗时、四个固定种子canary及16份零抢占metrics均本机复核。四个canary文本相同不替代新Math。候选首轮出现1条_index_linearized_jit_function调优记录，后两轮为0；控制三轮均0。

最终持久化的路由快照（非精确最终调用总数）记录3328次准备，2872次融合、456次回退，其中2650次B64融合，证明真实服务确实使用快路径。候选position和req_indices均int64，原KV slot mapping调用保留。未支持的prefill、混合、重排等继续原方法。

原件：/workspace/benchmark-results/20261016_decode_batch_meta_service_tianshu。归档6271893bytes，SHA305d49c78728134571f2993148de6b5351fd90b76c336f669c268fcdc0e39acd。复核：`python3 experiments/tianshu_decode_batch_meta_20261016/verify_service_downloaded.py --root <下载目录>`。

当前正式最佳4093.81 total tok/s、818.76 output tok/s、TTFT4309.35ms、Math102/105保持；本次为小规模筛选，非N256正式成绩。接续Scheduler/KV只读诊断，不重复页长32等已测负/平结果。
