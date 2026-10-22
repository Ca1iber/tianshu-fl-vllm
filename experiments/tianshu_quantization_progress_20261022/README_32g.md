# 天数量化探索，2026-10-22

本轮已完成：[最终报告](FINAL_REPORT_32g.md)。

新正式4K **5777.22 total / 1155.45 output tok/s，TTFT3459.44ms**，较旧5695.95吞吐+1.43%，TTFT降低5.58%。全新C8与真实C64 Math500 Level3各101/105=96.1905%；C64实际M4096原生INT8全42层、M64融合Graph与LM64 INT8均通过。

[直接Bash推荐入口](../tianshu_native_int8_prefill_release_v2_20261022/README_32g.md)；[正式四轮报告](../tianshu_native_int8_prefill_official_results_v2_20261022/REPORT_32g.md)；[独立双并发质量](../tianshu_native_int8_prefill_quality_results_20261022/REPORT_32g.md)；[同期N64对照](../tianshu_native_int8_prefill_results_v2_20261022/REPORT_32g.md)。

C8与正式程序原审核失败状态均保留，分别按实际BF16覆盖和原命令/CSV中的workload metadata完成独立CPU恢复，未修改原raw。完整范围、hash和零抢占检查见报告与gate。新16K未复测。代码/报告/关键结果在天数GitHub，原始数组、cache和全部日志仅容器。

[第一阶段负结果](../tianshu_quantization_survey_20261022/REPORT_FINAL_32g.md)仅属于当时的W4/W8原型阶段；后续原生INT8新阳性证据纳入本最终报告。原模型SHA保持不变，旧发布版和全部历史记录保留。
