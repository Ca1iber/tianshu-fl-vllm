# BI-V150 32G：等价 magic 解码减少寄存器，但仍更慢

三档全部256码点 CPU/GPU等价，8场景的64项全GPU FP32参考、640项CPU所选向量、相同Graph改变Q/K/V/table/lengths/slots及完整storage/输出/segment/RNG保护通过。实际字节格式、V缩放、K编码、writer、attention数学、tile/warp/stage/segments不变，仅改变V读取helper。

|Batch/上下文上限|K8V16 ms|magic V 1/32 ms|耗时增加|
|---|---:|---:|---:|
|1/4096|0.036029|0.042491|17.94%|
|1/5120|0.040300|0.048278|19.80%|
|8/4096|0.088812|0.125423|41.22%|
|8/5120|0.106290|0.152772|43.73%|
|64/4096|0.589351|0.860462|46.00%|
|64/5120|0.728903|1.073527|47.28%|

以上同进程9次交替event，每组20次热Graph，包含一次KV写入和一层decode。计时在改变数据后的第二阶段，实际长度4089/5113；并非完整模型服务吞吐。其他缩放与两个ragged/strided场景也没有收益。原始event、全部三档结果和编译元数据见DECODE_GATE_32g.json及decode_timings_32g.csv。

B64寄存器142→155，均无spill；相比上一写法的183已下降，但性能没有兑现。因此不能把上一版负收益完全归因于寄存器/占用率。这里只观察编译资源与耗时，还没有硬件counter结论；停止推进这两个固定参数V8变体到服务，下一步用容器自带IXplorer做小范围硬件profile。

正式最佳仍是K8V16/default4096的5101.45 total tok/s和同版本Math102/105。没有修改最佳启动器、模型或基准逻辑，本轮没有prefill、完整allocator、模型、服务、Math或正式benchmark结果，不采用。完整原始结果与冻结来源留/workspace/benchmark-results/20261020_v_int8_magic_decode_tianshu；GitHub仅存代码与关键JSON/CSV/报告。初次GPU预检查的路径歧义及修正见PRECHECK_NOTE_32g.md，失败日志保持原样。
