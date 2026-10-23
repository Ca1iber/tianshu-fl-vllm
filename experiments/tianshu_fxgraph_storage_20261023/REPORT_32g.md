# 天数容器可重建FX图缓存空间恢复

本轮42层W4组件启动前，GPU已空闲。容器磁盘只剩约283MiB；本次只清理`/tmp/torchinductor_root/fxgraph`下3379份可重建FX图缓存，约4.1GiB。没有删除模型、输入、原始评测输出、冻结控制器、Triton二进制或已引用生成源码。

先将全部FX图缓存归档到容器`/dev/shm/tianshu-fxgraph-20261023.6JFt2F/fxgraph_backup.tar`（4364625920字节），计算归档SHA256，并以`tar --compare`与3379文件逐个SHA256复核，然后才删除原缓存普通文件；原目录结构保留。此归档位于容器内存文件系统，是可重建缓存的临时恢复副本，不是需要传到GitHub的原始实验数据。

清理后重新核对当前候选934模型文件、95份实际GPU质量控制文件及5份CPU质量审计恢复文件，全部SHA256不变；固定384项缓存检查通过，无恢复或缺失；已采用r3启动器CPU dry-run通过。可用磁盘恢复到约4.4GiB。没有在模型benchmark计时期间执行维护。

原始维护证据仅在容器`/workspace/benchmark-results/20261023_fxgraph_storage_tianshu`。关键结果和证据清单见同目录Git记录的`STORAGE_GATE_32g.json`，归档及缓存不上GitHub。
