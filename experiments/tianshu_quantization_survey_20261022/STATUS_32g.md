# 2026-10-22 量化探索状态：第二轮进行中

每天北京时间06:00–次日00:00可用，00:00–06:00不运行GPU，本日23:58硬停。正式发布仍5695.95 total /1139.19 output tok/s、TTFT3663.77ms，官方C8 Math103/105、实际C64补充101/105。新候选未接模型，不借旧质量。

第一轮295矩阵×5格式扫描、9版300配置/900变化相位与A8倒数修正已完成，详见 REPORT_FINAL_32g.md 与 FINAL_AUDIT_GATE_32g.json；没有可替代现用路径的新候选，原始checkpoint未改。

第二轮A8布局：14配置×8shape，24个变化输入artifact、每实现43316121个元素的码/scale由独立CPU全部逐位验证。M2048最佳单量化器19.908→17.081us(+16.55%，绝对2.827us)，M64仅2.938→2.900us；没有模型收益/新Math，未采用。ROWS分组没有胜出，warp变化的小收益不能直接外推。见 ../tianshu_a8_launch_geometry_20261022/REPORT_32g.md。

原生INT8支持：M8被M>16约束拒绝，V1失败保留；M64两K/两布局、12相位完整CPU INT64和Graph支持通过。真实M64 GateUp 24候选/72变化相位完整INT32/投影/输出位相同且独立CPU通过，但最快NN0.066165ms仍慢于现用0.063122ms，不采用。见 ../tianshu_native_int8_gateup_20261022/REPORT_32g.md。

当前继续原生INT8大M128/512/4096预填充：严格用现用逐M配置与实际binary作控制，2布局×2epilogue行分组，完整量化/GEMM/中间读写/缩放/SiLU计时。已冻结源并通过12点离线编译，正在GPU与独立CPU审核。最大单INT32 workspace约201MB，尚未审核模型allocator/bank成本，不提前宣称收益。目录 ../tianshu_native_int8_prefill_20261022。

源码、报告、关键结果天数GitHub研究分支；完整raw/日志/profile仅容器，无本地项目副本。不改原模型、不改架构、不训练、不动沐曦仓库或用户已有未提交文件。

更新：大M V1因debug源码路径导致完整cubin不匹配，零计时case，已保留；独立来源复现+V2继续严格全binary校验，三个M全部GPU/独立CPU通过。NN/ROWS1完整GateUp：M512 0.261038→0.232177ms(+12.43%)，M4096 1.957708→1.493048ms(+31.12%)；M128负。优先NN复用现有bank，下一步固定NN/ROWS1的42bank含Down holdout，再决定真实服务与新Math。尚无新模型成绩，正式发布不变。详见 ../tianshu_native_int8_prefill_v2_20261022/REPORT_32g.md。
