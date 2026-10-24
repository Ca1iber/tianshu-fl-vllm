# 已结束运行的可重建 FX 图缓存维护

2026-10-24，GPU空闲时仅清理 /tmp/torchinductor_root/fxgraph 普通缓存文件。先在容器 /dev/shm 保存完整tar、全文件SHA及逐文件tar比较，所有检查通过才删除原缓存文件。保留模型、完整实验数组、原始输出、Triton binary、固定qualified_model_cache和全部源码。恢复用 tar -C /tmp/torchinductor_root -xf ARCHIVE_32g.txt 所列归档。归档属于可重建编译缓存的容器内临时备份，不是GitHub实验数据。
