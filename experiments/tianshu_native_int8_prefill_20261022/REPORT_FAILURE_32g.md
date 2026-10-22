# V1控制cubin不匹配：零数值/计时结果

M128第一次严格完整binary SHA检查失败，complete=false、cases=0，原源/raw保持。相同Triton缓存key不包含源码路径：原资格debug路径large_tiles/kernel.py，新默认缓存large_tiles_v2/kernel.py。CPU解析ELF证明.text/.rodata/.note全部ALLOC内容及属性一致，仅DWARF等非ALLOC不同；不能将此检查失败当性能负收益。

后续独立来源诊断按两个AST相同设备来源复现六次，M128/V1、M512和M4096/V2各自完整cubin严格符合正式gate。新V2合并其索引并冻结实际group引用和cache文件，仍要求完整binary SHA，不降低数值/性能门槛。
