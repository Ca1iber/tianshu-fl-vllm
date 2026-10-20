# 受控冷采样缓存与四类ABI入口预热

实际LibEntry.key对N/seed/offset的do_not_specialize参数只保留整数类型，N在本模型B1..64范围均为i32。原生philox helper从两个有符号int64状态字读取seed/offset，因此入口只有seed i32/i64 × offset i32/i64四种组合。B37冷调参恰好发生在offset跨2^31后；192次不同B调用不等于192个入口，也不能证明64个ConfigCache条目。

V2用私有CUDA generator在代表B64下覆盖四种入口，检查目标entry确实存在，保留全部既有entry及配置，默认CPU/GPU RNG不变。CPU实际调用原生key函数3584次覆盖64批和signed-int64边界，证明四类key和N值无关。

后续GPU受控实验只在隔离进程把该原生tuner的配置/benchmark缓存替换为内存测试夹具，保留所有原库/源/默认调参协议；不删除用户SQLite配置。先测offset跨界时B37冷调用，再保留已选低位入口、预热V2，检查跨界调用和全B/四类入口无新调参、原RNG保持及可比较路径完整字一致。测试夹具不进入服务或正式成绩，旧图/模型warm点数保持。
