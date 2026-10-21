# 2026-10-21 10:42 北京：TopP 基线纠正，GPU 空闲

继承预约至 2026-10-22 00:00，GPU 硬停 2026-10-21T15:58:00Z。当前推荐 407 source/62 env、正式 4K5586.28、Math102/105 不变。用户询问最近方向，已说明框架预算、投机输入/接受率、采样组件三方面，以及这轮错误基线没有新增收益。

431-source 新服务控制臂在 policy.py:27 因 Router.fn 不存在失败，零 benchmark 轮，未运行候选。原 GPU 实验目录/CPU_GATE/PLAN 已冻结，不能修旧目录或单纯重启。API/Engine 已退出，GPU68MiB空闲。新 CPU 复核报告在 tianshu_topp_baseline_review_20261021。

重要纠正：推荐版实际上已 w16/s1 + GEO 32K/16K，生成 SHA09156e66...。plain installed w4/8K 对比的组件 28–35% 耗时减不是新增成果。上一份 TOPP_COMPONENTS_DONE 状态和 NEXT_TOPP_SERVICE_PLAN 的行动建议失效；不要继续解包 Router 把已优化 GEO 绕回原生8192。

旧 TopP 预筛选已经跑完且主场景负收益（读 REPORT 而非旧 README）；旧6144/4352也已跑过。后续先检查已归档结果避免重复，再选真正新假设；若同预算用新 INT8 精度栈复测，明确这是交互验证而非首次方案。

GitHub推送已确认HEAD6bd5ac7。两处用户dirty不动，Ops4cdc3b7不变，无本地副本、无子agent。SSHmaster04:27建立，16:27前刷新。Goal继续active，每轮报告，观察超时不重启，优先小4K，不常规16K。
