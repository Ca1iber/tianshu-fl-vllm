# 历史不可变数值数组按完整SHA共享，保留全部评测证据

所有当前GPU/质量/短测/组件及其CPU复核队列均已终止，GPU空闲。为下一模型质量/性能验证留空间，扫描benchmark-results内至少1MiB的数值bin/npy/npz等原始文件；排除source/controller快照、checkpoint、code与cache目录。相同device/size/权限/owner/group/完整SHA的不可变数组只在原始数组之间硬链接，保留所有原路径/字节/原始manifest，不把活跃源码、模型或cache当目标，不丢弃结果。

每文件原始SHA及最终每独立inode的SHA完整复核，全清单仅容器；当前944模型/控制器、384严格cache、prefill组件21项输入及LM16缓存前后SHA核对。算法/计划在执行前冻结，所有数字收益来自实际statvfs差值。没有GPU计时或模型改动，代码、门槛和报告GitHub。
