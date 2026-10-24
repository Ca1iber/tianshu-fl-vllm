# 已校验SDK捕获副本的独立备份

3个SDK独立捕获副本已压缩并持久保存在容器，SDK_RAW_ARCHIVE_32g.txt记录大小、SHA及tar逐字节对比结果；SDK_ARCHIVE_MEMBERS记录精确成员。完整归档不上传GitHub，仅上传本说明和元数据。当前模型及CUDA计算没有变化。

若容器重启导致内存盘副本消失，在容器内恢复原路径：

```bash
mkdir -p /dev/shm/tianshu_QKVO64_model_profile_CPU_20261024
tar -C /dev/shm/tianshu_QKVO64_model_profile_CPU_20261024 -xJf /workspace/benchmark-results/20261024_QKVO64_immutable_SDK_snapshot.tar.xz
```

模型prepare仍会按IMMUTABLE_SDK_REFERENCES中的逐库SHA验证捕获副本，并核对已安装SDK相同来源哈希；只恢复捕获副本，不更换已安装运行库。普通模型代码、量化参数和评测输出独立保留，不把SDK引用当成代码文件的独立副本。
