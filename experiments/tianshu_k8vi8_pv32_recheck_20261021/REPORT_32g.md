# PV32完整attention数值通过，当前配置显著负优化

独立复核五个case×三阶段×三配置共45条路径、468个CPU FP64参考向量，以及真实writer的271872个V整数元素。完整物理缓存、输出/scratch保护区、输入与CPU/GPU RNG检查通过。真实writer、量化、producer和reducer全部计入CUDA Graph，每组七次轮换计时、每次二十重放。

|batch/context|K8V16 ms|原生V8 ms|PV32 ms|PV32相对K8V16耗时|
|---|---:|---:|---:|---:|
|1/33|0.016576|0.018240|0.023306|+40.60%|
|8/257|0.021299|0.024286|0.029275|+37.45%|
|40/4096|0.502578|0.670998|1.043518|+107.63%|
|64/4096|0.540265|0.751463|1.450511|+168.48%|
|64/5120|0.668212|0.936236|1.805201|+170.15%|

K8V16和原生V8采用T16/1warp，PV32采用T32/2warp；结果属于这些完整配置，不能只归因于整数指令。本次C64/4K实际n_regs分别176、189、198，三者均n_spills=0，未观察到spill。没有底层计数器或细分时间证据，不断言寄存器、动态P量化或硬件指令是唯一瓶颈。

真实8448-byte K8V8页含K/V和两个FP32 scale平面，KV写入与读取同一原生storage；组件尚未接入完整engine allocator或prefill。虽然数值通过，但当前完整组件性能已明显落后，不投入新Math或整模型benchmark，未采用。不能把本结果扩大为所有整数PV方案均无收益。

## 复核器单行错误恢复

GPU probe完整结束，原CPU后处理遗漏了已移除PV64的layout检查，原runner保留failed与KeyError。此目录复制原verify_gpu.py，仅移除这一个过期检查项；VERIFIER_FIX.diff与VERIFIER_SOURCES.sha256明确记录。新复核器独立验证全部原始冻结源及快照，复用原GPU数据，无GPU重跑。GPU_GATE、ROUNDS与RECOVERY是本次恢复复核结果，不覆盖旧失败记录。

原始根目录：`/workspace/benchmark-results/20261021_k8vi8_pv32_attention_v2_GPU_tianshu`。原实验代码在[PV32组件目录](../tianshu_k8vi8_pv32_attention_v2_20261021/README.md)。四配置首测的PV64短序列超差另行保留，不属于本次通过的三配置结果。
