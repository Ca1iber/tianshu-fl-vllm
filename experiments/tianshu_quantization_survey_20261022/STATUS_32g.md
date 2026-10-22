# 2026-10-22 量化探索状态：继续第二轮

每天北京时间06:00–次日00:00可用，00:00–06:00不安排GPU，本日23:58硬停止。用户恢复量化探索后继续以下两项，不把第一轮阶段性收尾视为不再探索。

第一轮：295个矩阵×5格式CPU扫描；9版权重组件300配置、900变化相位及A8倒数修正完整GPU域与6种shape对照完成，均未找到超过现用路径的候选。原checkpoint、冻结源与raw结果的最终审核见 FINAL_AUDIT_GATE_32g.json，总报告 REPORT_FINAL_32g.md；这些历史记录保持。

第二轮A：保持旧A8 FP32 RN除法/RNE不变，只改每CTA行数和warps。42组CoreX离线编译已通过并冻结源码；8种shape×14配置、变化输入/全输出CPU参考与Graph轮换计时正在运行。目录 `../tianshu_a8_launch_geometry_20261022`。

第二轮B：CoreX原生torch._int_mm真实M64、NN/NT、变化输入Graph支持已验证；M8因接口要求M>16拒绝，V1原失败保留。新真实GateUp组件已冻结并通过12个epilogue离线编译，等待上一GPU任务完成；量化+原生INT32 GEMM+独立scale/SiLU均计时，现版BM64/BN64/BK128/W4 two_dot和actual binary作控制。目录 `../tianshu_native_int8_gateup_20261022`。不得根据接口支持推断性能阳性。

正式发布仍 **5695.95 total /1139.19 output tok/s，TTFT3663.77ms**，官方C8 Math103/105、实际C64补充101/105；新候选均未接模型，不借用已有质量成绩。原模型不改、不转换架构、不训练。代码/报告/关键数据存天数GitHub研究分支，完整raw只在容器，无本地项目副本。
