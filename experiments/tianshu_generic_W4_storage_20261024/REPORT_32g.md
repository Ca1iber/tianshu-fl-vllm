# 标准W4实验前的可重建缓存空间恢复

2026-10-24，GPU空闲后，仅处理/tmp/torchinductor_root/fxgraph的86个普通缓存文件，约101MiB。该目录无symlink、无.py/.so/.safetensors，也未出现在r5的997项模型来源清单中。未处理FX目录外生成模块、模型、rawarchive或固定384二进制缓存。

先以新SHM目录保存完整tar、逐文件SHA清单和tar--compare结果，再验证原文件所有SHA；通过后删除这86份可重建缓存，目录保留。997项r5源码SHA及固定384缓存前后检查均通过，restored=0/missing=0。工作盘可用由82MiB恢复至181MiB，原始证据/归档位置与SHA见EVIDENCE_32g.txt。备份在容器SHM，属于可重建缓存恢复副本；不是模型和评测rawarchive。
