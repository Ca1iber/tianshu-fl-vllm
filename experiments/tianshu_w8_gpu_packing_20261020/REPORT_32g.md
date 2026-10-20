# 43组W8静态权重GPU打包组件

未改现行5375.50正式运行时。复用已通过整数GEMM组件的原始量化kernel b9b692cf561a35ef9c38d05d1edf7aa804436728923efa4800f914ff6b656152，按8192行分块；42个GateUp和LM head全部真实checkpoint权重，3次交替顺序，共129次完整CPU/GPU q与scale逐字节相同。全部43组原BF16/INT8/scale哈希与新100/105 Math实际模型权重一致，独立CPU重算全部完整payload哈希。

另9组变更输入测试覆盖全零、RNE边界、符号/尺度变化、行stride、K259/K33尾和8193行chunk尾；完整值、输入位、保护区和CPU/GPU RNG均通过。没有CUDA Graph：这是一次性加载工作。

两臂明确统一4个CPU线程，计时均含原BF16 D2H、finite检查、打包/分配/传输与三组完整SHA证明；GPU臂还拷贝q/scale回CPU。逐bank三次中位数加总：原CPU12490.945ms→GPU8489.354ms，减少32.0359%。这不是原实际worker133.413秒线程配置的严格对照，也不是模型完整启动或吞吐收益。实际服务集成、实际CPU线程数/模型内存accounting和启动时间仍待核验。

原始目录/workspace/benchmark-results/20261020_w8_gpu_packing_GPU_tianshu；完整模型和量化payload不上传Git，仅保留代码、关键JSON和报告。
