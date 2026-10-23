# r4后历史快照去重

GPU空闲时，仅将内容、大小、权限、uid/gid完全相同的已结束source_snapshot/controller_source_snapshot等历史副本合并硬链接。1847个目录、364556文件，新增合并65027份；全部原路径/字节SHA在操作前后逐一核验。未链接或修改实时源码、模型checkpoint、模型数组或评测正文。释放3850133504字节（约3.59GiB）；完整20MB逐文件清单仍在容器，SHA及关键门禁见SNAPSHOT_DEDUP_GATE_32g.json。没有性能或数值算法改动。
