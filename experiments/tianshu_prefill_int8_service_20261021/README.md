# 大 M prefill/混合批次 INT8：模型4K配对

两组固定相同NN纯decode M8/M64、相同42层逻辑权重与物理布局；候选只把启动时已预热、128对齐且contiguous的大M GateUp矩阵改为A8/W8。其他M/布局委派原BF16路线，LM不拦截。大M可能包含prefill与decode混合行，候选对矩阵所有行使用同一精度。43-bank总payload仍1,326,938,112字节；内存计量、全部逻辑SHA由父安装和审计验证。NN decode对照本身不是当前正式推荐版本。

选定分块来自已通过96配置/288阶段的grouped组件，M129全部负收益因此保留BF16。中间对齐M选最近上方实测配置；两组在父warmup/capture完成后预热同一实际M集合，用全部A量化及真实layer0 W选中INT64→BF16检查，再激活请求。CPU/GPU RNG与CPU线程数保持。未知M或布局回退，不在请求中首次编译新INT8形状。

CPU配置门407运行来源、62环境、原13参数。独立审计仅显式适配父BF16 prefill实际覆盖条件，候选必须有42层实际INT8源码、分块、二进制与M4096请求证据；完整保留其他原生/权重/Graph/FutureKV/抢占检查，不用startup dummy伪造请求。

未计分seed canary分别使用不同cache_salt，使相同prompt/seed两次都完整重算前缀；计时请求不带seed或盐。每组canary必须复现，跨组允许因prefill精度而变。策略不读取seed/盐/内容决定路线。先4K/1K C64 N64三轮排除首轮；若收益明确，再全新105题Math严格>95%和正式N256 4K。没有把组件收益写成模型成绩，不做常规16K。

代码、报告、关键JSON/CSV上传天数GitHub，完整原件只容器。北京时间2026-10-21 23:58硬停止。
