# v1 冷启动对照：重复 SQL 钩子失败，保留原始结果

native 三轮 4K/1K C64 N64 为 5135.46、5308.63、5296.90 tok/s；均为 64 成功请求。首轮排除后的实验均值约 5302.76，仅作诊断。原始结果保留 `audit_failed`，不采纳为新的正式成绩。

原 loader 实际为 56 CPU 线程，43 bank 打包 126.03 秒，load_model 129.47 秒，API ready 192.74 秒。完整内层 W8A8、图捕获/重放、KV 调度与环境审计复查通过；外层失败原因是新装的 early SQL 包装在真实请求中调用为零。

真实链路 `/experiments/tianshu_sql_workflow_20261014/hooks/sitecustomize.py` 会覆盖该方法，并执行既有 `reflect_one.py`。实际 worker 的 `sql_reflection_records/route_85096.json` 已确认 `reflection_scope=one_requested_table`。因此 10 月 20 日 SQL CPU 对照只重复验证了已采用优化的正确性，不能再计作现有服务的新收益。

v2 移除重复包装，各模式继续使用同一既有目标表反射，只比较默认 56 线程 CPU、4 线程 CPU 和 4 线程 GPU 打包。v1 源码、零调用记录、三轮数据和失败状态均保留。
