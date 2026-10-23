# 2026-10-23 Tianshu量化进展（北京时间16:03更新）

正式最佳仍是r3：4K 5950.75 total /1190.15 output tok/s、TTFT3174.58ms。以下均为组件结果或候选，不是新的正式分数。

| 真实投影/方案 | 当前对照→候选 | 组件速度变化 | 质量状态 |
|---|---|---|---|
| 42层Down64，AWQ RTN W4，CUDA Graph | 2.320458→2.188805 ms/42层 | +6.01%，省0.131654ms/解码步 | 完整2359296 FP64格通过；模型未集成 |
| 42层Down64，dotsaddz RTN W4，CUDA Graph | 2.320545→1.940144 ms/42层 | +19.61%，省0.380401ms/解码步 | 完整2359296 FP64格通过；正集成 |
| LM64，AWQ RTN W4，相对当前W8A8核心eager | 0.611353→0.535017ms | +14.27%，省0.076336ms | 完整3phase数值通过；未集成 |
| LM64，dotsaddz RTN W4，相对当前W8A8核心eager | 0.611353→0.519834ms | +17.61%，省0.091519ms | 两W4共50135040 FP64格通过；未集成 |

Down用42份不同真实层权重轮换，含完整eager/graph poison、输入/权重/scale/zero版本/全字节/guard/RNG校验；LM对照是实际采用的b9b692自写W8A8核心，偏向对照地预分配其q/scale/output，没有错用未采用的原生INT8或BF16。采用7轮交替/旋转顺序、排除首轮后6轮中位数。详情在对应组件REPORT和INDEPENDENT_CPU_GATE，不把组件百分比当整机收益。

物理格式已GPU独立确认：generic INT4四个常量例全部精确；非恒定AWQ测试8格式有2个精确；dotsaddz30格式唯一匹配自然N uint4打包与BF16加性zero=-logical_zero*scale。每个格式固定完整INT64参考16384输出，全输入guard。运行时RTN保留原checkpoint，不做模型转换/微调，也未做AWQ校准。

模型优先只改Down64，LM保持r3，便于归因。v1（4efb8be）权重加载/预热/内存探针通过，但W4 capture0/旧BF16capture42，startup拒绝，零题评测。v2（998971f）通过公开register_kernel更换实际原custom-op CUDA实现，绑定原_init_fn/bank closure、原schema/fake不变，其他M委托真实原backend；916模型来源保留903原来源、85控制器、63环境/13官方argv。先C64新鲜105题，严格>95%至少100/105，再C8，然后才短4K；任一失败不做原样重跑追分。

v2原始状态目录：`/workspace/benchmark-results/20261023_dots_Down64_v2_math64_tianshu`；质量runner PID58990，API59773（启动时记录）。时段截止2026-10-23T15:58Z（北京时间23:58）会终止。模型代码冻结后不得直接修改；修复须新版本保留原证据。此进展文档不属于冻结模型/控制器来源，可继续更新最终状态。

另有CPU group32/64/128/256误差研究，3层3fixture均完整FP64：平均投影L2 9.79/11.15/12.53/13.97%，仅供后续独立候选，未测新group GPU和模型质量。当前运行仍为G128。此前prefill RMS/QKV融合虽C8/C64均101/105，但短4K只有+0.0507%、波动1.0088%，未采用、未跑正式，不与本W4混用。

所有代码/报告/关键结果推天数GitHub分支exp/tianshu-quantization-20261022。全部大数组/原始日志/模型/缓存只在容器，无用户本地项目文件。MuXi只读参考，没有修改其仓库。

15:45现场状态：v2实际模型启动和独立startup审计已通过，真实W4 M64 capture42、原BF16 M64 capture0、42个实际worker-memory-profile scratch探针完成。已进入C64完整105题，完成101题，剩4题生成中；尚无最终准确率，C8和短4K均未开始。不会把组件通过/预热命中当作模型正确率通过。

16:03更新：新鲜C64已完整103/105=98.095238%，独立CPU全量复核通过（916模型来源/85控制器/105题与全部raw SHA/实际profile逐字重算）。104正常stop、1个max_tokens，按官方review全部计入分母；无请求错误。C8同一冻结版本已开始评测，最后确认42/105，尚无最终分数。完整模型质量资格仍pending。

下一步已准备并冻结e74e039短4K控制器 `experiments/tianshu_dots_Down64_short4k_v2_20261023`，CPU预检确认精确903来源r3对916来源新W4、13官方argv/63环境仅入口不同。真实GPU执行强制要求独立完整C8/C64 QUALITY_GATE合格；每臂N64x3/exclude1，筛选>0.5%收益、candidate spread<1%、TTFT<=1.01。当前未运行短4K。新鲜质量全部完成后运行独立结果目录的verify_cpu.py生成完整QUALITY_GATE，再执行run_pair.sh；不在已执行目录修改任何来源。

CPU另发现G32非对称minmax的3层3fixture平均投影误差7.70%（当前G128对称12.53%）；新group/参数未GPU或模型测试，保持独立候选，不混入当前模型。仅代码、关键误差表与CPU门禁入GitHub。
