# BI-V150 32G：指数随机数入口类型预热准备

从V1的实际LibEntry.key行为出发，只预热四种seed/offset的i32/i64组合；N在当前词表与B1..64均为i32，不以64种N等同64个调优入口。全程fork_rng保护默认CPU/CUDA状态，只用33,423,360byte暂存，显式禁止Graph内预热。调用实际FlagGems set_philox_state建立类型边界，结束恢复原状态。

独立私有FlagGems DB/代码缓存；不清共享编译缓存。计划所有64种N的后预热调用、原33组全局/混合/独立generator逐token与RNG检查，另24组seed/offset跨2^31边界的B1/8/64检查，要求后续不新增指数调优。GPU尚未运行，没有新服务、质量或吞吐结论，不自动加入5101.45最好。新版本与V1原失败严格隔离。
