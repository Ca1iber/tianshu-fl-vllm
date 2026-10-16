# V3：控制性能请求完成，追加收尾诊断失败

这次不是完整A/B，也没有候选成绩。控制的C64/N64 4K/1K三轮分别3866.554、4033.396、4039.473 total tok/s，首轮排除，后两轮均值4036.435。192个性能响应usage/输入种子/完整文本/流计时全部保留并下载重算，三轮新增调优均0，7份工作负载metrics均0抢占。

失败发生在吞吐后的额外1-token诊断客户端assert。没有序列化该失败请求的原SSE，不能确定具体哪个检查不满足；不作模型数值、路由或质量推断。after_metrics尚未取得，round3_after_metrics真实存在。保留原phase=failed、日志和70份源快照，不补造after_metrics，不修改冻结run/hook/policy/审计器。

独立verify_failed.py只对已完成工作负载取证：在独立namespace执行冻结audit_runtime，将末尾指标文件读取唯一替换为真实round3_after_metrics；报告明确限定范围，不将诊断阶段视作通过。全部70源快照及本地仓库所属源SHA、原serve/default2048/256seq/75warm/40环境、CV/KV16/S8/三M2048/strictTopP32Kgeo及192响应均通过。新候选未运行，无新Math/正式基准/采用。V4另建目录移除该不必要诊断，不能覆写此次失败。

原始归档8,663,240字节、34分片，SHA256 f18ff0a11bcd4834c991cca58f3644c5f968624cfac5c8f4e8fce7517760b5c8。原件位于service-results/20261016_tianshu/prefill_cadence_failed_v3/20261016_prefill_cadence_v3_service_tianshu。
