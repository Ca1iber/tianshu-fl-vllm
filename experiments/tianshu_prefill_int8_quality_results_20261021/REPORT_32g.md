# 新prefill INT8 Math105：102/105，通过>95%

2026-10-21 06:42北京时间完成，全新同一407运行来源/12主控制来源/62环境/原始13参数的105题Level3。**102/105=97.142857%**，105条预测和review按index 0..104全覆盖，原始API无错误；全部逐题分数、input/output/total用量、原行SHA独立重算。temperature1/top_p.95/max_tokens32768、eval batch8、timeout3600保持官方要求。

104条stop、1条max_tokens，命中上限的样本保留并计入105题，没有删去失败样本；共355190输出tokens。完整回答、原始review和HTML仅容器，GitHub保留逐题评分/用量/哈希与关键审计JSON。旧版101/105没有复用，也不把随机采样下本次102题正确写成统计显著的准确率改进。

实际C8 GateUp NN整数/LM BF16路径、新prefill INT8真实请求42层均验证；Math实际触发的新prefill M为256，不声称它验证了C64/M4096所有请求。24个对齐M预热全部A量化和选定真实W的CPU INT64→BF16核验、CPU/GPU RNG不变通过；Math自身新prefill和纯decode均不改变checkpoint/原BF16权重。额外43 banks共1326938112bytes，真实DeviceMemoryProfiler计入，新增prefill无额外packed W payload，actual KV40731 blocks，零抢占、FutureKV全部释放。

正式4K使用同一407来源/12主控制/62环境/argv，逐一核验，自动进行原脚本N256×4轮排首轮。它尚未完成时不报告新正式分数；C64路径由正式运行的实际执行审计确认，C8 Math不冒称C64准确率测量。结果目录的verify_quality.py是额外CPU归档器，复用冻结runtime audit但独立复算全部105条raw评分/用量/index，并记录自身SHA。

原件 `/workspace/benchmark-results/20261021_prefill_int8_math_tianshu`。启动包 `../tianshu_prefill_int8_release_20261021` 通过CPU干运行，含26份生成依赖的原字节，不含权重或缓存。小测5425.62→5520.64（+1.75%）仍是N64配对数据，正式结果另存 `../tianshu_prefill_int8_official_results_20261021`。

07:02后同一profile正式4K已完成且独立复核采用：5586.28 total/1117.25 output、TTFT3780.63ms，见 ../tianshu_prefill_int8_official_results_20261021/REPORT_32g.md。本质量archive保留当时official starting_server的观测，不改写原始阶段记录。
