# K8/V16 attention线程块坐标顺序：CPU通过，GPU待测

基于当前合格S16 reader（SHA7b94b63e…），尝试让两个KV头的线程块在grid线性布局上更接近，检验页读取局部性是否有变化。硬件实际调度和收益不能从坐标顺序直接推定。所有方案保留原线程块数、浮点计算、tile、warp策略和reducer，只在3D producer反向映射program_id；2D路径维持原坐标。

|方案|新grid轴依次表示|
|---|---|
|current|query, head, segment|
|heads_first|head, query, segment|
|segments_first|segment, query, head|
|heads_segments_first|head, segment, query|

source.py固定原件SHA、逐字可逆替换，并验证AST只改变3个program-ID赋值。CPU_GATE_32g.json的260组坐标集合检查证明双射、数量和2D回退一致；56组通过实际合格Producer类的假kernel调用，验证仅launch grid改变，原tile/warp/stage等参数一致。CPU未执行Triton，不是GPU数值或吞吐成绩。代码不安装模型hook。

GPU准备沿用上一组件真实12KiB物理页、K软件8/V BF16、原writer、分段/launch策略和Graph检查，4方案×12场景×2份变数据。覆盖B64的4K/5K与ragged/strided，以及B40长上下文和B41跨8192边界；这些是毫秒级单层组件，不是完整16K benchmark。预计96份全输出FP32参考、原始完整物理缓存及scratch guards检查；保存全输出和选定逻辑K/V，verify_gpu.py离线重算256个CPU参考向量、逐位比较4方案，并核查实际grid和432组交替计时。完整payload仅留容器，GitHub只保存代码、报告与关键JSON/CSV。

run_component.sh是待执行入口，GPU空闲后才允许启动，硬截止北京时间2026-10-20 23:58，单次上限1800秒。目前不排入已运行队列，优先完成未来KV正式16K和已排队RoPE/KV融合。若组件有实测收益，再做4K N64服务对照；不会为无收益组件追加Math或完整benchmark。
