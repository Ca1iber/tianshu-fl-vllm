# 完整整数PV Attention：数值参考闭环，性能明显退化，不采用

2026-10-21，BI-V15032G，现行407源/62env/13原serve参数对应的K8V16 reader f1df731b为真正对照。七路：当前K8V16、原V8、PV默认false、整数PV T32/T64各w2/w4。十场景、三阶段，在真实原生页表/存储、writer、producer和reducer上捕获与重放，共210条GPU路径完成。单层40/16K仅组件覆盖，不是完整16K benchmark。

|C64 context|当前K8V16 ms|原V8 ms|整数T32/w2 ms|整数T64/w2 ms|整数T32/w4 ms|整数T64/w4 ms|
|---:|---:|---:|---:|---:|---:|---:|
|4096|0.539900|0.752026|1.450207|1.749446|2.637803|2.646427|
|5120|0.668410|0.935839|1.805371|2.165115|3.310645|3.300507|

七轮交替顺序、每轮20次热Graph，包含真实writer+量化/scale+attention+reducer。4K最快整数PV耗时比当前增加168.61%，比原V8增加92.84%。当前176regs/0spill，T32w2为198/0、T64w2为256/36；实际元数据不能单独证明退化根因。V8物理页8448B（含两组scale），当前12288B，减少容量没有带来速度收益。

独立CPU重新加载保存的原BF16 Q/K/V与实际输出，共2184个参考向量，其中1104个独立INT64 PV算法向量，302592个实际writer V量化值和302592个K编码字节；三阶段输入变化、稀疏/strided页、负slot、raw cache/output/scratch守卫和RNG检查保持。42项默认false/真实2D回退完整GPU输出逐位相同；CPU保存选中输出再次核验。

PV采用分tile P*Vscale重新量化，声明整数算法参考与量化误差界通过，但相对理想V8 attention的原容差有两项失败，全部保留，不能宣称精确attention或模型质量通过。没有接入模型，没有新Math或正式吞吐成绩，推荐仍5586.28/Math102。

原CPU verifier数值检查后在C128启动元数据断言失败：2D实际segments=1而非传入但未使用的16。原冻结代码/失败保留。独立CPU恢复仅修正该谓词，未重跑GPU、未改容差；最终GPU_GATE/ROUNDS/KEY_TIMINGS位于 ../tianshu_k8vi8_pv_full_cpu_recovery_20261021。GPU原件 /workspace/benchmark-results/20261021_k8vi8_pv_full_GPU_tianshu，原始PT和完整日志仅容器；本报告和关键JSON/CSV进GitHub。
