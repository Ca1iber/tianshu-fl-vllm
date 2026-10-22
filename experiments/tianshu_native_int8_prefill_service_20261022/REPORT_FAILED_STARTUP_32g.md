# 首次接入启动失败，未进入模型评测

固定版本 c1a59d2，原始记录位于 /workspace/benchmark-results/20261022_native_int8_prefill_service_smoke_candidate_tianshu。
EngineCore 初始化时 native_policy.install 创建 native_prefill_records 失败，FileExistsError。继承环境中该名称已用于 TIANSHU_NATIVE_PREFILL_PROBE 的 attention 探针。没有加载模型、执行吞吐或精度评测，不形成成绩。

完整失败状态保存在 GPU_FAILURE_GATE_32g.json；原数值源码保持不变。修复放在独立 service_v2 目录，新 INT8 探针使用 native_int8_prefill_records，继承的 attention 探针目录和环境保持原样。
