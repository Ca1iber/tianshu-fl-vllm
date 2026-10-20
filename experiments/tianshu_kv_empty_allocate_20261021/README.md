# 原生FullAttention KV空分配快捷路径CPU证明

只针对已验证单池FullAttention、blocksize16、caching开、非EAGLE、metrics collector关、131072上界；运行Request、每次一个token、无新prefix/投机/外部KV/encoder/延迟cache/全序列准入/预留块。已分配页涵盖computed+1，且num_cached_block已涵盖min(computed+1,request.num_tokens)的完整块时，native remove_skipped/allocate_new/cache_blocks各阶段均无工作。candidate返回native.empty_kv_cache_blocks同一对象，其他输入调用原函数，native源码不改。

check_cpu用真实原生KV管理器和Request，42层同一组、逻辑uint8/12KiB页元数据。20条trace含事件开/关、prefix1/15/16/17/63/64/129/4096/16384/131040、页边界、异步computed领先confirmed输出、free与prefix重用；完整manager/coordinator/unit/pool/块hash/refcount/空闲链表/事件/Request字段核对。1065个快捷返回经过状态比较。满池仍能接受15个已有页token；需要新页时15次原生None保持。

所有执行GPU不可见，无GPU上下文、GPU Tensor或内核，也未分配物理KV数据。7轮正反交替，每轮2000次CPU空分配，原生5.6488775us→含guard/统计的候选1.2808145us，不是模型吞吐或GPU时间。

check_guards另用19个真实API/回退/异常场景：默认/has_scheduled_false/多token/非法0token/prefix新增/显式空blocks/lookahead/external/delay/encoder/admission/reserved/预留耗尽/WAITING/PREEMPTED/位置上界/partial prefill/未知keyword/非法Request。API签名、返回值及原生空singleton、异常类型和文本、整个管理器递归pickle、Request完整递归字段值均一致。

首次主CPU夹具用prompt8构造满池，原生不登记0个完整prefix块，故guard正确回退、shortcut计数不等预期；改为prompt24、31块池，15请求各2页且已有1完整prefix块，保持guard不变。首次API夹具直接比较不同Request只读view对象的identity，已改为完整递归字段值比较，管理器pickle/原生异常检查不变。两次失败原件与SHA保留；通过结果分别CPU_GATE与CPU_GUARD_GATE。

服务单独见tianshu_kv_empty_allocate_service_20261021，不能用本CPU证明替代实际模型收益/质量。
