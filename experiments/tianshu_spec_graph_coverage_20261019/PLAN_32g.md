# 天数投机 Graph 覆盖率诊断草稿（2026-10-19）

隔离脚本已镜像至天数，CPU 透传检查与真实容器的 CLI 导入通过，尚无 GPU 诊断或优化收益。先完成小批次 Top-P 服务对照。

旧 ngram 固定 gather 后仍比近期非投机控制低约 34%，但实际 async=False、75 点补齐和 hybrid guard 关闭，不能把全部退步归因验证算子。源码 uniform 条件要求所有请求 Q2；混合 Q1/Q2 及全 Q1 不满足。当前尚无请求阶段的实际缺图比例。

拟在原 ngram/fixed-index/正 FP32 标准 rejection 配置上添加纯观察器，保留所有决策和返回值。只在真实 execute_model 上建立计数上下文；外部 profile/capture 计入独立计数，分别记录 scheduler 查询长度、CPU prompt 完成状态、实际 dispatch 描述符与 FL GraphWrapper 成功调用走的分支。existing_graph_replay 仅指匹配 mode 且调用前已有 graph 的成功原生调用；不等于逐 kernel trace，也不产生 GPU 时长。Python observer 不在被重放图内部。

每轮前写显式 phase marker，轮后以单独标记 canary 触发上一阶段完整落盘；吞吐只作诊断负载时间，观察器开销未测不能报作候选比赛分数。所有模型请求正文/usage、源码、环境、75/hybrid/async 实际状态均需留存；marker 读取与 JSON 落盘成本明确保留。

下一步先校验真实 FL GraphWrapper 源和调用控制流、完成原生签名边界 CPU 检查、接入隔离 hook 与两轮 4K/512 诊断，硬停沿用 2026-10-19 23:58+08。只在实际数据显示全 Q1 缺图占比明显后，再设计 Q1 独立描述符和捕获方案；不可仅 force_uniform=True 复用 Q2 图。
