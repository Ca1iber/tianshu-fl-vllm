# 50GiB容器磁盘配额恢复，原始评测保留

DownV2预填充服务于编译阶段遇到Errno122 Disk quota exceeded，尚未进入评测。原raw目录、模型源与控制源保持原样；磁盘满导致原result.json也未成功写出，不能把该目录标为complete。后续用相同冻结V2代码、全新raw根重试。

只对归档source_snapshot/controller_source_snapshot的同内容文件做物理硬链接去重，所有原路径与文件字节保留，未链接到可变的运行源码或实时GPU缓存。涉及1,448个快照目录/178,222个文件，全部SHA256前后逐文件检查通过。CPU准备副本约省363MiB，其余归档约省2.2GiB。

另将/tmp/torchinductor_root/fxgraph可重建编译缓存约4.6GiB完整备份到容器/dev/shm，再经tar --compare验证后清空；确认冻结890模型源与27控制源均不引用该缓存路径。该备份只用于本次容器会话，后续无需备份时也可按原软件流程重建编译缓存。模型、Git仓库、原GPU输出/日志、root/.triton及root/.cache/vllm均未清理。

可用overlay空间由约76KiB恢复到约7.4GiB。r1原384缓存文件全部检查通过（0恢复/0缺失），原发布启动器CPU dry-run通过，未启动GPU。r1正式5777.22 tok/s不变。完整checksum清单压缩件及维护日志只在容器/workspace/benchmark-results/20261022_storage_dedup_tianshu，GitHub只存关键检查和报告。
