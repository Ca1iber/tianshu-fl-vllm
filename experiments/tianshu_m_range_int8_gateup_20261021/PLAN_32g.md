# GateUp分M布局/融合扫描

24 M点1..4096、真实layer0 Gate/Up权重，同一A8/W8量化与已有模型权重签名。分别计A8量化、NN投影、SiLU、总耗时；NT/NN同tile布局对照；现用NN总链路对比分组双dot融合BM64/128、BN64/128、GROUP1/8共8配置。当前M64已经融合，current_total必须为现有BM64 BN64 W4 G1融合；M8和预热24个M使用真实当前unfused配置。其余M只探索INT8子系统，不冒充模型基准，不直接扩大精度白名单。

每case三组动态输入/权重符号与scale变化，全输出BF16位相同、全INT32和投影BF16位相同、完整量化精确，CPU选点INT64精确+激活.001+.02abs，guard与输入不变。只two_dot，禁已知失败single_dot。7轮交替顺序×20graph热测。保存CPU fixtures和实际GPU选点，完整输出在线检查，不称CPU重载完整输出。W静态NN转置和当前已存在bank复用分别记；不增加A pack。范围白名单需后续邻点/42层缓存/模型验证。源、计划、门槛GPU前冻结，修复新目录，90分钟或23:58北京硬停。
