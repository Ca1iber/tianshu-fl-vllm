# 现行paired-K固定T16：更多warp或stage均明显变慢，保留w1/s1

2026-10-21，BI-V15032G，原当前407源绑定、reader f1df731b和真实12KiB K8V16页面/writer保持。只n>24改变warp/stage，全部T16、分段/.cv/输出dtype不变。七配置×14场景×两阶段，共196完整GPU参考与全部输出14077952元素的离线重查，304个独立CPU向量/2128项原记录CPU参考通过。所有新配置输出逐位等于当前；完整缓存字节、output/scratch guards、负slot、RNG和同Graph变Q/K/V/table/length/slot检查保持。

|T16 warp/stage|C64/4K ms|C64/5K ms|4K regs/spill|
|---|---:|---:|---:|
|当前1/1|0.552635|0.682612|176/0|
|2/1|1.323351|1.645418|132/0|
|4/1|2.016669|2.521479|121/0|
|1/2|1.634649|2.036230|141/0|
|2/2|1.994093|2.479449|122/0|
|1/3|2.237259|2.791454|128/0|
|2/3|2.387253|2.974786|116/0|

九轮正反交替、各20次热Graph，882组计时，包含真实KV更新+decode+reducer，未含参考计算。较少寄存器并未带来速度收益；全部无spill也不能独自解释差异，不据此编造occupancy或硬件原因。n1..24相同配置的亚百分比差异只作噪声；全部候选和原始顺序保留，不能挑低batch波动当成功。

旧Producer硬编码num_stages=1，故本轮在最终JIT调用前应用stage2/3，并由4928实际启动参数CPU stub与GPU实际metadata/二进制hash独立确认。原Producer/Reducer源码AST不变，CPU stub不冒充GPU编译证据。原10-19未配对K7b94/c6源的stage/warp结果不同于本f1df源，本轮补齐组合后同样不采用。

单层40/16K只是组件覆盖，不是完整16K评测。length1捕获后4K和8193的同一Graph也完成。没有新服务/Math/N256，当前正式5586.28/TTFT3780.63/Math102不变。runner236233已Z，GPU空闲。GPU_GATE与GPU_ROUNDS保留全部关键证据，PT/完整日志/生成源快照仅容器 /workspace/benchmark-results/20261021_pair_t16_pipeline_GPU_tianshu。
