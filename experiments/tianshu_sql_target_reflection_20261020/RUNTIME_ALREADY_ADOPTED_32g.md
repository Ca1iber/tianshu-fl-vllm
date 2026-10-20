# 实际服务已于 2026-10-14 采用同类目标表反射

10 月 20 日的 CPU 对照完成了全部 1158 张表、43784 行和特殊模式的独立数据一致性核验，单次冷目标表 API 的 4.57 秒到 12.48 毫秒测量仍是有效组件数据。

继续追踪真实服务后确认：现有合格配置通过 `tianshu_sql_workflow_20261014/hooks/sitecustomize.py`，在 FlagGems SQL 模块加载后安装 `reflect_one.py`。该 helper 同样只反射请求表。真实 worker 记录 `reflection_scope=one_requested_table`，而新加 early 包装计数为零，被既有 hook 覆盖。

结论：CPU 实验是既有服务优化的补充正确性证据；不能把 4.57 秒到 12.48 毫秒再次算为当前正式版本新增启动或吞吐收益。后续冷启动对照保留既有 helper，只改变加载线程与静态权重打包执行位置。
