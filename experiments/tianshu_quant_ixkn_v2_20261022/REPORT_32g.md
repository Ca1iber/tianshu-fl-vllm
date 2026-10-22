# 直接CSV采集与原生BF16对照

V2候选退出0，完整候选输出在采集前后与原独立审核artifact逐位相同，编译binary SHA相同，CSV共105行、8个section。V3原生BF16的父timeout wait status为0、应用前后selected输出与原artifact一致；CSV103行、8section，实际调用_iluvatar.ops.mm.mm_out。CPU重新解析CSV、核对kernel/PID/source/参考时延和关键字段。没有把SDK预估的所有122项都当作可用项。

|SDK报告字段|W8A16 NT旧候选|原生BF16|
|---|---:|---:|
|kernel时延us|225.876|56.038|
|grid blocks|16|32|
|threads/block|256|1024|
|vector regs/thread|115|25|
|shared bytes/block|49152|65536|
|TCU efficiency|16.0942%|30.9701%|
|DRAM read efficiency|16.6465%|62.5142%|
|achieved occupancy|7.30%|25.53%|
|no eligible|84.57%|65.97%|
|executed instructions/warp|23561|3500|

单独用同一fixture读取libtuner实际best_config：原生BM64/BN64/BK128、16warps、2stage；旧W8候选BM64/BN128/BK128、4warps、1stage。SDK提示小grid和shared occupancy限制。不同算法/布局/tile的综合对照不足以证明单一根因，原生共享内存更大也更快，不能简单说“shared大所以慢”。

据此设计新12点组件：包含原候选控制、与原生相同tile/warps/stages，以及邻近8/16warp、BN/BK组合。测量才能决定是否改善。该新轮在../tianshu_w8a16_native_tiles_20261022，尚未完成前不声称收益。所有时延为组件/kernel，不是模型吞吐；正式发布5695.95 total tok/s保持。

原始CSV/日志仅容器，Gate保存必要metrics及SHA。首次导出失败单独保留，未作为计数器证据。
