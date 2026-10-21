# 多查询 SplitKV 第一轮：19 场景有效，完整计划未通过

2026-10-21 BI-V150 32G。原18源冻结组件共计划26场景，已完成19个长序列/参差查询场景后，在q2_short_k2的control_2D第0相位遇到严格FP32参考失败：max_abs=.0091307163、3个元素超出既定.002+.01*abs(reference)。原GPU任务phase=failed、complete=false。该短场景候选尚未执行；后续6场景未测，不能称整轮数值门通过。源、CPU gate、原失败结果全部保留。

独立CPU部分证据复核仅覆盖已完成19场景：266次完整保存输出对FP32、760个独立CPU注意力向量、105238784个原BF16 K量化编码值、931条交替Graph计时，以及完整18源/快照和实际2D/3D启动记录均通过。见verified/GPU_GATE_32g.json与KEY_TIMINGS_32g.csv；其中complete仅指部分证据复核完成，component_plan_complete=false明确不代表26场景通过。

| C64/4K query width | direct byte 2D ms | best of 6 SplitKV ms |
|---:|---:|---:|
|2|23.733623|.882030|
|3|23.926511|.892537|
|4|23.872363|1.625536|
|8|23.838658|1.792682|

仅单层attention读KV热Graph，不含writer、模型GEMM、采样、draft/rejection或服务成本；6选1探索结果。基线为当前byte大BM256/Q32/T128/w16直接回退。当前请求能放进8192pages arena时走gather+dense，不在本轮对照内，因此不能把该倍数当作模型吞吐提升，也不能代表旧DSpark的完整BF16路径。C64/4K现有arena确实不足，但投机接入尚未实现，现用407源模型明确SpecNone，不能简单开开关叠加。

下一步先核验短序列控制为何不满足理想FP32参考。原生代码在PV前执行P.to(V.dtype)，本身采用BF16概率乘法；这只是待验证原因，未据此放宽误差阈值或宣称候选正确。再补当前gather+dense与旧投机小查询BF16读取的实际对照。当前最佳正式5586.28/TTFT3780.63、Math102/105保持不变，无新模型成绩/Math或16K。
