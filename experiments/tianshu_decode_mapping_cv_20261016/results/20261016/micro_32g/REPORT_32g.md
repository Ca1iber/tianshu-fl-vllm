# 当前CV配置直接映射：B64反而慢，不迁移旧收益

Oct12旧配置曾约-1.3% attention耗时，不能直接当当前CV/KV16/w1/S8版本的收益。当前仅改producer映射与grid第0维，完整保留原attention循环、softmax、int64指针及reducer；正确处理0/1内部空query。

|B/ctx|strided|内部空query|原 ms|候选 ms|耗时变化|
|---|---|---|---:|---:|---:|
|1/33|False|False|0.017797|0.017626|-0.964%|
|8/4096|False|False|0.140237|0.149453|+6.571%|
|24/4096|True|False|0.411525|0.344731|-16.231%|
|32/4096|False|False|0.502501|0.546982|+8.852%|
|64/4096|False|False|0.845309|0.910001|+7.653%|
|64/5120|True|False|1.052726|1.137318|+8.036%|
|8/257|True|True|0.023083|0.022427|-2.840%|
|64/4096|True|True|0.773271|0.605931|-21.640%|

B64满批4K/5K慢7.653%/8.036%，不接服务。B24和人工内部空query有组件收益，但不能据此推出C64真实服务收益，未做选择性routing或正式/Math。原网格的空CTA本来早退，减少网格/二分查找不等于按网格比例减少KV读取；未采硬件计数，不将差异断言为occupancy或HBM分区原因。

8场景、16个原始/改变QKV阶段，包括改变空query位置后重放同一Graph，32项全输出FP32与176个CPU head参考、所有输出/分段尾部检查通过；两方案全部输出逐位相同。112个原始事件组（每组20 replay）、中位数/差额、grid/warps/strides、64源快照及本地所属源已下载独立核验。两处可逆修改可重建生成源、reducer AST相同，未保存全部GPU Tensor，本地核验是记录与出处。

归档4,346,833字节、17分片，SHA256 81b9f8d1511ad963cb8045174bf9b3c6b37a54f75becc10b8933b9651cb4ad6e。原件service-results/20261016_tianshu/decode_mapping_cv_micro/20261016_decode_mapping_cv_micro_tianshu。未改最佳4093.81/Math102。
