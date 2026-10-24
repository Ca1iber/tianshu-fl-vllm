# QKV+O64 拟合W4核心实现，模型尚未就绪

2026-10-24，准备下一版组合候选。保持原r4模型、原13项serve参数、prefill/GateUp/Down/LM/KV/采样。只在FULL uniform64 decode、连续BF16[64,2048]、无LoRA/spec时将42层QKV和42层O替换为原生G128拟合affine W4；其它路径逐项交回原生attention op原CUDA backend。CPU拟合源分别为已独立核验的O和QKV权重研究，固定6轮，完整payload必须匹配后续GPU合格组件的全SHA。

本目录目前只准备policy、纯CPU量化helper、scope与hook；没有最终SELECTED_BANKS/CPU_GATE、candidate_profile、完整质量控制器或新模型GPU运行。CORE_CPU_GATE仅验证6个代表银行与CPU已验证payload全字节一致及范围守卫，不是模型就绪、GPU数值或质量门槛。必须先完成现有O64 C8+4K，再做QKV拟合GPU资格，之后才能补齐全部84银行选择/冻结源、运行新鲜C64/C8 Math105>95%和4K端到端对照。O64的102分不能借用给此组合。

新增84组三个buffer共210,567,168字节，两模式均保留相同payload及原BF16/W8。在真实DeviceMemoryProfiler内加载/原生worker峰值窗口内预热全部84银行，并验证全模型唯一存储被统计；不手动扩KV容量、不声称净显存节省。现有模型/冻结源未改，当前GPU不运行该候选。GPU截止2026-10-24T15:58:00Z，所有代码/报告/关键证据GitHub，完整数据只留容器。
