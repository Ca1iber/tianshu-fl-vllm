# BI-V150 native Decode / FA2 Prefill

本目录是天数独立实验，不修改沐曦源码或官方benchmark。

- probe_decode_segments.py / probe_decode_tiles.py：真正的原生3D Decode sweep；保留失败配置。
- profile_service.py：4K真实服务profile；no-graphs仅诊断，不作为计分。
- validate_hybrid.py：经过native builder/forward、CPU边界、padding与CUDA Graph的独立FP32检查。
- run_service_comparison.py / run_service_series.py：官方serve参数、两组五轮16请求4096/128短A/B。
- run_math_eval.py：独立MATH-500 Level3，官方生成参数，105项完成后审计报告。

默认关闭的VLLM_FL_ILUVATAR_HYBRID_ATTENTION：保留Native纯Decode，FlagGems FA2只处理Prefill；mixed batch在标准reorder threshold1基础上拆分。单卡BF16 dense MiniCPM，无量化、LoRA、投机、多模态或DCP。不适用时保留原版。性能服务显式设置FLAGGEMS_ILUVATAR_COMPACT_VARLEN=1，未启用之前失败的全FA后端。

独立7场景数值检查通过，原版部分Prefill误差也有记录。Pure Decode与未重排fallback位级一致。短服务原版3024.12→4547.85 total tok/s（+50.39%），TTFT10251.90→6089.29ms，KV容量两侧521840。不冒充官方成绩，不沿用原版数学准确率。当前独立MATH运行中。

run_service_series.py 支持独立指定 --output-tokens、--concurrency、--requests、--rounds，保留原默认16路/16请求/128输出/五轮。下一组使用4096/1024、64路/64请求/三轮，首轮不计入实验均值，仍不属于官方256请求成绩。

完整报告与原始数据位于workspace service-results/20261011_2100_native_attention_tianshu。GPU测试串行，只清理自有进程。用户已明确续约到2026-10-12中午12:00，预约JSON截止更新为11:58；此前已启动的数学评测仍保留旧23:58 watchdog，完成后不会延用它。
