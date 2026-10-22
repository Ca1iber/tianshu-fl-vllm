# 原生INT8预填充 V2：复现资格缓存来源，保持完整binary匹配

V1在M128完整cubin SHA检查停止，零数值/计时case，原源/失败不覆盖。cache key相同但debug路径来自large_tiles/kernel.py或large_tiles_v2/kernel.py，M128全部ALLOC ELF section一致。CPU首次想直接复用旧缓存，发现M512/4096旧cache未保留预期cubin；原audit_cache_cpu.py与CPU准备失败另存。

独立GPU编译诊断已验证V1/V2设备函数AST相同、6次零输入无RNG变化，并复现三个严格资格binary：M128取V1来源，M512/4096取V2来源。V2仅把这三个匹配缓存的索引合并到独立qualified_cache；实际group引用的来源文件及复制索引全部hash冻结。GPU仍要求完整binary SHA等于正式gate，不放宽控制，也不改旧cache。新的候选独立编译。

数值算法、配置与计时继承V1：真实第0层W8签名，M128/512/4096，NN/NT两布局×epilogue ROWS1/4(BLOCK1024/W8)，3变化相位，完整INT32/BF16投影/输出匹配、独立CPU INT64与全量投影hash、同一Graph/RNG/guard。7轮×20replay包含A8、原生INT32中间读写、缩放/BF16投影和同SiLU。每相位只存共同INT32，raw仅容器，关键报告/源码/JSON/CSV推天数GitHub。

原checkpoint只读，不改架构/训练。最大单workspace201326592字节，不据组件宣称模型显存或质量已合格；阳性再42bank holdout/真实allocator/fresh C8+C64 Math/4K对照。每天06:00–24:00、23:58硬停，单任务30分钟，冻结后修复新版本。
