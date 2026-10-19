# K8/V16完整attention组件（未接入实际engine）

接续十种decode组件、K8/V16 B64 4K减时28.6783%的原始证据。新目录补真实TritonAttentionImpl.do_kv_cache_update/forward、metadata builder、mixed/prefill/native BF16 staging、bounded容量回退与大B FULL decode兼容。21场景BF16对K8/V16成对，4K服务或质量尚未运行。

42个真实FullAttentionSpec保持原128维几何，dtype carrier UInt8并将page_size_padded设12288，原生reshape按照真实12KiB页计数。K以byte读取，V从原始V地址创建共享存储的BF16元数据view（不复制内容），引擎未来在初始化时注册，组件在warm/capture前创建并缓存；捕获时缺失view即报错。Writer采用此前已通过穷举编码和完整物理检查的实现，write与read都取同一V view。prefill只解码K、V按BF16复制到已计入的共享arena。模型/Q/out BF16，原RNG与原serve argv保持。

CPU已跑1280策略/7容量/67K dense地址、42不可变spec转换、实际Producer 3D/大batch2D、4种mixed dispatch路径、18个packed/strided V视图与3错误元数据拒绝。真实GPU仍须验证所有逻辑值/物理字节、变数据同Graph、CPU/GPU参考、scratch/padding/RNG及native allocator CPU fixtures；不能把CPU准备或此前decode成绩当新pipeline/engine结果。完成后再安排真实ModelRunner内存预算、35图启动、服务A/B和全新Math。
