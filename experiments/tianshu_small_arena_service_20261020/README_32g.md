# 512→128MiB prefill暂存区服务对照：准备完成

两组同为当前已验证K8V16/4096/83预热点/严格TopP32K/M2048/长上下文S16，原始serve argv、KV watermark0、max_num_seqs256不变。唯一运行配置差异是复用prefill解压arena容量32768→8192页；模型cache物理布局、解码算术和RNG保持。完整原始暂存调用与超容量fallback继承不变，具体GPU边界和限制见[组件报告](../tianshu_small_arena_20261020/README_32g.md)。

COMPOSITION_32g.json核验236份相同源、48项环境中只有容量模式不同、相同原始serve命令，并用前一控制组实际记录复查原生FL Graph审计。新hook只在bounded_prefill模块导入时装容量子类，长S16链运行一次，原engine在determine_available_memory内执行真实reserve。服务审计核验实际scratch字节/表/调用栈、物理KV容量、原生Graph捕获/重放及所有请求。

run_screens.sh串行4K N64每组3轮排首轮，再在4K回退不超过1%、冻结源码一致且时间充足时执行16K N64同规格对照。此处16K针对KV容量压力；不是完整N128官方benchmark。未取得实际服务收益前不跑新Math或完整正式测试，当前发布版本不含此实验。

15:11结果：4K N64 -0.01403%、16K N64 +0.65787%，两负载共768/768成功、两组真实预留/同源/Graph核验通过。16K两组均6次抢占，收益较小，不采用、不单独跑新Math或完整benchmark。详见[结果目录](../tianshu_small_arena_results_20261020/README_32g.md)。
