# 全注意力单池decode空KV分配快捷路径

保留原Scheduler初始化及FutureKV准入、释放和pop包装；只替换FutureKV controller.original这一个内部委托。运行请求每次仅1 token、无prefix新命中/投机/外部KV/encoder/延迟cache/全序列准入/reservation，且已分配页足够、没有新增完整前缀块可登记时，候选返回原生empty_kv_cache_blocks同一个对象。其余参数与错误交给原原生方法。两组相同guard/统计，control一直调用native，candidate才提前返回。

原生allocate_slots源码不改，但候选会跳过已由CPU证明无副作用的调用段。既有FutureKV记录中的native_calls字段描述委托调用数量，包含新的快捷路径；不能再把该旧字段解释为每次都执行了native函数体。新记录单列calls/native_calls/eligible/shortcuts，要求委托总数一致、最终计划释放、零准入拦截。

原生CPU管理器/Request、42层同组12KiB spec、前缀cache、blocksize16、maxlen131072验证20条trace，包含事件开关、页边界、位置上界、异步computed领先confirmed tokens、free及前缀复用。完整pool/块hash/refcount/空闲链表/事件/Request状态与空结果singleton一致；满池15次原有页空分配与15次新页None均一致。另19个API/回退/异常案例以整个管理器递归pickle与Request完整字段值核对。无GPU可见、没有物理KV分配。

两组继承359来源合格模型父版，62环境、原13项serve参数，4096预算/maxseq64/async；不合并配置缓存、日志阈值或新采样预热。4K/1K C64 N64 3轮首轮剔除，真实GPU收益另判；不是Math或正式N256/16K。截止北京时间2026-10-21 23:58。
