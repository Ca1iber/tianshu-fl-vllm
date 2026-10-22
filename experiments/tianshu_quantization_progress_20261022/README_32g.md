# 天数量化探索，2026-10-22

当前已交付推荐版：[最终报告](FINAL_REPORT_32g.md)。

新正式4K **5777.22 total / 1155.45 output tok/s，TTFT3459.44ms**，较旧5695.95吞吐+1.43%，TTFT降低5.58%。全新C8与真实C64 Math500 Level3各101/105=96.1905%；C64实际M4096原生INT8全42层、M64融合Graph与LM64 INT8均通过。

[直接Bash推荐入口](../tianshu_native_int8_prefill_release_v2_20261022/README_32g.md)；[正式四轮报告](../tianshu_native_int8_prefill_official_results_v2_20261022/REPORT_32g.md)；[独立双并发质量](../tianshu_native_int8_prefill_quality_results_20261022/REPORT_32g.md)；[同期N64对照](../tianshu_native_int8_prefill_results_v2_20261022/REPORT_32g.md)。

C8与正式程序原审核失败状态均保留，分别按实际BF16覆盖和原命令/CSV中的workload metadata完成独立CPU恢复，未修改原raw。完整范围、hash和零抢占检查见报告与gate。新16K未复测。代码/报告/关键结果在天数GitHub，原始数组、cache和全部日志仅容器。

[第一阶段负结果](../tianshu_quantization_survey_20261022/REPORT_FINAL_32g.md)仅属于当时的W4/W8原型阶段；后续原生INT8新阳性证据纳入本最终报告。原模型SHA保持不变，旧发布版和全部历史记录保留。

推荐版作为冻结检查点保留。后续继续探索其他M范围的原生INT8收益，全部新尝试使用独立目录，未通过新模型与质量门槛前不改推荐版。本索引允许更新，独立版本的最终报告和数值源码仍冻结。

## 14:40 后续量化研究（r1推荐版不变）

- [32点M曲线](../tianshu_native_int8_m_curve_20261022/REPORT_32g.md)：真实三类基线，32点/96阶段独立复核。多数中大M组件正向；4K主路径4096已经采用，因此未声称新的整机提升。
- [LM64原生INT8](../tianshu_native_int8_lm64_20261022/REPORT_32g.md)：全部8配置数值逐位一致，最快仍慢约0.3%，反量化抵消GEMM收益，不采用。
- [Down单层](../tianshu_native_int8_down_20261022/REPORT_32g.md)及[全42-bank MLP](../tianshu_native_int8_down42_20261022/REPORT_32g.md)：候选完整MLP在M64/512/4096分别+8.55%/+13.28%/+20.47%，504份阶段CPU独立复核通过；这是新Down近似，尚无新的模型质量或服务吞吐。
- [新Down服务设计](../tianshu_native_int8_down_service_20261022/PLAN_32g.md)和[新质量验证](../tianshu_native_int8_down_quality_20261022/PLAN_32g.md)：890模型源、27控制文件冻结，原877源完整保留；原13argv/63环境。新增约504MiB单一NN packed Down bank计入真实显存profile。14:38启动新scopeall C64完整105题，之后才跑新官方C8；若质量不足，另测预先声明的prefill-only范围。当前运行结果只在容器，未宣称通过。

正式r1仍为5777.22 total tok/s、TTFT3459.44ms，fresh C8/C64各101/105；原不可变FINAL_REPORT_32g.md与r1标签不变。
