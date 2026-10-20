# 沐曦最新提交只读复查

2026-10-21。直接将 GitHub 网络字节转发到天数容器已有 bare 参考仓库，没有本地源码副本，没有改动沐曦源码或其远端。

- flagos-contest 当前分支 ed3bcfe6da2704d696b35c6ec8048afcd1d7cced；新增 10af519 原生 top-k/top-p 接入，主要修复沐曦布尔计数 INT32 和编译控制流，替换其原排序 fallback。天数已使用同类原生 pivot 筛选，且另有严格概率质量与阈值精度优化；不能把沐曦相对排序 fallback 的提升挪作天数预期收益。沐曦报告还明确保留了阈值/并列输入差异。
- flagos-ops-lib 当前分支 8eb57f17d5a3a93f7399af440d8dc319b0916c00；新增只反射目标 SQL 表。这是冷配置读取改动，其提交也未宣称稳态吞吐收益。核对当前天数原 359 源码 profile 与审计后，确认其中早已包含 tianshu_sql_workflow_20261014/reflect_one.py，且运行审计要求实际 worker 执行 one_requested_table。无需重复移植。

本次读取的容器参考位置 /workspace/reference-repos/metax-latest/{flagos-contest.git,flagos-ops-lib.git}，更新的是 origin/exp/packed-bundle-service-20261007 参考指针，未移动本地已有 heads。结论仅对本次查阅提交与当前天数 profile 成立。
