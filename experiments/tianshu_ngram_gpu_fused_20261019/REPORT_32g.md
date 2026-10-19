# GPU ngram 两阶段整数搜索：组件有效，尚未接入服务

2026-10-19，BI-V150 32G，执行版本409ad75。保留原始完整历史搜索、最长匹配优先、同长度取最早位置、禁用行与空结果约定。候选用两个整数Triton kernel，替代原生已编译GPU搜索；没有改变标准拒绝采样或接受率。

CPU公式检查32839例通过。真实GPU的6场景各测原始与改变后的输入，共12次数值检查，候选eager、改变输入后的同一CUDA Graph、原生提案和CPU参考的整数结果完全一致。包括非整齐批次、短上下文、131072长度边界、无匹配、禁用行和有步长的int64输入；有步长场景明确使用原生forward参考，其余使用真实构造后的编译kernel。输入与RNG状态保持，未序列化完整GPU张量。

|组件负载|原生已编译GPU搜索 ms|两阶段候选 ms|
|---|---:|---:|
|N63，约4K上下文|1.259897|0.013417|
|N64，约5K上下文|1.256773|0.015675|

以上为7次交替GPU event的中位数，每次10遍CUDA Graph replay；共28条event。双方都包含同样的组件调用边界，不代表模型整轮吞吐。原生仅约1.25ms，而原生GPU投机服务筛选仍只有2507.66 total tok/s，低于当前非投机配置，因此不凭组件倍率采用，不借用现有Math分数。

初版探针在构造原生Proposer时缺少当前vLLM配置，零GPU样本退出；失败归档保留。独立probe_gpu_v2.py恢复真实配置与推理上下文，未覆盖旧探针或旧结果。

完整成功原始包4559606字节，SHA-256 `4fef155a15addb20ae25a09dbf15b080642c4e00fbc24ffe0cfd0ccfd032c8f0`。本地独立核验63份源码快照、12次数值检查与28条event通过，退出后GPU空闲。失败原始包4554337字节，SHA-256 `b351cf9f3cae80ae8c237f30ef13390c78bc98fc6f1cfa858c7c9d2feb8ffebe`。原始包位于工作区service-results/20261019_tianshu/ngram_gpu_fused_micro_v2与ngram_gpu_micro_context_failed。

没有候选服务测试、新Math-500或正式benchmark。当前正式最好4093.81 total tok/s、Math102/105保持。
