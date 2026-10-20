# V1 启动中止记录

2026-10-21 01:50（北京时间）主动中止 V1 control。尚未执行 benchmark 轮次，未产生可报告的吞吐结果。

原因：早期捕获检查固定读取旧 `launches` 字段。融合候选真实调用写入 `fused_launches`，因此该客户端检查会误拒绝候选。发现后在 control 启动阶段停止整个队列，保留原始冻结源码、state、server.log；不修改已启动版本源码。

修复位于独立 `tianshu_int8_mlp_fusion_service_v2_20261021`：candidate 检查 fused_launches，control 检查 launches。GPU 融合 kernel、配置和精度策略保持一致。V2 两臂重新运行，不复用 V1 的任何性能数据。

容器原始记录：`/workspace/benchmark-results/20261021_int8_mlp_fusion_service_control_tianshu`；停止原因另存在对应 runner 的 `STOP_REASON.txt`。
