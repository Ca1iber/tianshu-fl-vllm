# GateUp补齐独立服务对照

基于5482.79正式合格配置。两臂都保留LM小批次BF16/64整数、BF16预填充linear_contiguous、gpu4静态bank加载、成对K与Future-KV，只有TIANSHU_TILE_PADDING_MODE不同。

candidate把GateUp M8..64补到32/64；M<8保留原BF16。每个custom op分配目标q/scale/out，补齐写入在实际RNE量化kernel中，返回有效行view，不缓存共享custom op输出。M64沿用原量化器和原整数GEMM，LM路径不变。支持其他小M会改变其原BF16计算，若值得采用必须新Math。

组件已有26场景/78变数据相位逐位/CPU独立通过。这里另外审计真实CUDA Graph capture/replay、逻辑/实际M、指针对齐、分配shape/字节、返回shape、实际调用source及43银行签名。记录capture前后实际torch allocated/reserved，未重置或更改原profiling。

先官方原serve参数的小4K/1K、C64、N64，每臂3轮排首轮；六轮同来源与除padding开关外同环境。原始输出/日志仅容器，GitHub仅代码、报告和关键计时/元数据。不默认跑完整16K或把小测当正式分数。
