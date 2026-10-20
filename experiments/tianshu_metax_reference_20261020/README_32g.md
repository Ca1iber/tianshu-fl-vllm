# 沐曦只读参考检查：2026-10-20

通过只读GitHub API和git ls-remote核对两个用户指定仓库的默认HEAD，未修改、提交或推送沐曦仓库，也未在WSL建立克隆。检查时天数新长分段Math在运行，未运行其它GPU任务。

- flagos-contest：`4cd3acb9ce1abe01bfdc6b773d4dec01963303a4`，北京时间10月20日10:44的SQL报告指针更新。它不修改推理插件计算逻辑。
- flagos-ops-lib：`8eb57f17d5a3a93f7399af440d8dc319b0916c00`，北京时间10月20日10:42，仅反射目标SQL调优表，生产源10增1删和一份CPU回归测试。提交说明报告C50064G稳态4K/16K变化仅+0.06%/+0.05%，明确属波动；主要节约配置读取冷路径。天数已有单目标表反射的同类处理，本次没有新的稳态算子实现可直接移植。

此前10月19日原生top-k/top-p采样提交`10af519dc5c0e9893844bfeb722bb59301d72b01`和两个INT32 sweep实验也再次出现在最近历史中。当前天数已使用自己的严格TopP32K几何修正，未把另一vendor的历史成绩、编译兼容修复或构造样本筛选差异视作天数收益；当前225份Math冻结源保持。

来源：[插件提交](https://github.com/Ca1iber/flagos-contest/commit/4cd3acb9ce1abe01bfdc6b773d4dec01963303a4)、[SQL提交](https://github.com/Ca1iber/flagos-ops-lib/commit/8eb57f17d5a3a93f7399af440d8dc319b0916c00)、[原生采样提交](https://github.com/Ca1iber/flagos-contest/commit/10af519dc5c0e9893844bfeb722bb59301d72b01)。以上性能数值是沐曦提交的报告内容，本次没有重新运行其benchmark。
