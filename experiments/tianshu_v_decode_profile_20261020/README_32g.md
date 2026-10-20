# BI-V150 解码硬件计数器：工具失败记录

2026-10-20 两次 IXplorer Kernel Analyzer 4.4.0 采集均失败，没有可用的完整硬件分析结果。不据此判断 occupancy、访存瓶颈或优化收益，也不修改原始报告补造设备信息。

基于已验证的 K8V16 reader，固定 B64/4096、同一逻辑 K/V、12KiB 物理页、tile16/warp1/segments4。首次正常结束时工作负载 finite 检查通过，但 profiler 退出发生 SIGSEGV；随后尝试官方 `--kill on`，退出发生 pybind11 GIL 错误。两份 SQLite 报告均只有原始事件，DEVICE_INFO 和派生 METRIC 数据为空。import 第一份报告返回255，报 `Unknown device name`。

PROFILE_FAILURE_32g.json 记录只读数据库行数。有限值检查不是新的完整数值正确性证明。原始报告、日志及输入留在容器 `/workspace/benchmark-results/20261020_v_decode_profile_tianshu`。未改动最佳配置，没有服务、Math 或正式 benchmark 成绩。
