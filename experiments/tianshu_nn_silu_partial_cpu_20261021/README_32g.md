# 原32组组件的4组已完成数据，CPU局部恢复

只核验原failed运行中已完成的4个M8/连续/two_dot配置。复制原独立verifier，仅调整原终态、预期范围、4项计数和结果命名；原所有权重行/量化/INT64/投影/激活/完整BF16输出/轮转时序/冻结源检查保留。原single_dot未保存完整失败输出，另GPU debug已独立重现；本脚本不重跑GPU、不改原result、不把其变成complete。

PARTIAL_GATE中的complete只指局部CPU核验完成，original_experiment_passed=false，incomplete_case_not_recovered保留失败项。不能用4项M8结果代替32组、M64或模型成绩。
