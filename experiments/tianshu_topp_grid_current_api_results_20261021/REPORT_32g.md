# 当前推荐父配置的TopP16K/grid2：API完整通过，待服务实测

2026-10-21，BI-V15032G。继承当前407-source/62-env prefill INT8推荐配置，API闭包418来源。GEO数学源码09156e66...逐字不变，仅扫描tile32K/16K→16K/8K、程序数min(B,16)→min(B,2×原grid)、64行持久scratch，并把有效B扩至8..64，保证新Math C8实际覆盖新数值路径。Sampler函数/alias、p/温度/RNG保持。旧10/19不同配置+.807%不是本轮成绩。

GPU300组mask（25分布×6p×2阶段）、33真实Sampler全局/部分/全部Generator、8次跨C8/32/33/64共存图生命周期、8fallback全部通过。输入和真实工作输出包括行步长130567的C8/C64。独立CPU从初始input+实际p+packbits mask重建600份完整FP32输出，共2879631360值，全部完整SHA匹配；重算22056条FP64请求语义。保留17条strict=false，全部与当前基线该行逐位相同；发生变化的600行全部满足原1e-6质量、最小支持集和排序条件。没有把等基线边界谎称严格通过。

|p=.95 BF16组件|原32K ms|候选16K/grid2 ms|耗时变化|
|---|---:|---:|---:|
|C64普通，阶段0|2.583823|1.932051|-25.23%|
|C64宽分布，阶段0|2.049315|1.369778|-33.16%|
|C64有步长，阶段0|2.609649|1.974971|-24.32%|
|C8普通，阶段0|0.610657|0.664438|+8.81%|
|C8宽分布，阶段0|0.478740|0.469438|-1.94%|

全部50项两阶段中位数见CSV；500个原始event，每项五次交替×10热Graph，双方含同样完整输入恢复，CPU存盘不在计时内。局部C64明显改善、C8普通回退，均不等于模型成绩。实际GEO源/编译binary/网格/scratch与有步长命中记录核验，原生TopK/no-filter/B1/2/4/7/65回退逐位保持。

无模型服务、新Math或N256资格，不修改5586.28推荐。下一步另新服务目录：原生memory_profiling_fl真实with内yield前等量预热两种路径和所有B，新增33MiB持久scratch及临时/JIT模块真实计量，不resetpeak。服务仍原13参数/62env，先小4K三轮配对，稳定收益才新Math>95与N256；不常规16K。

GPU/完整张量原件 `/workspace/benchmark-results/20261021_topp_grid_current_api_GPU_tianshu`，完整CPU复核 `/workspace/benchmark-results/20261021_topp_grid_current_api_CPU_verified_tianshu`。复核器在 `../tianshu_topp_grid_current_api_20261021/verify_gpu.py`。GitHub只有代码、报告、关键JSON/CSV，所有raw仅容器；原源和CPU_GATE已冻结，退出GPU空闲。
