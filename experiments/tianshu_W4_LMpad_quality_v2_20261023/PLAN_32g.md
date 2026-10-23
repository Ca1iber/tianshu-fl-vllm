# LM全小批次W4模型：fresh完整C8/C64质量

使用唯一最终模型`tianshu_W4_LMpad_service_v2_20261023`，全部source在首次GPU前冻结；source数量由CPU门禁实际计算（预期944），13项serve参数不变，完整63环境仅PYTHONPATH换服务hook。C8保留原FutureKV candidate，C64仅FutureKV control与原模型质量协议相同。C8和C64必须同模型/source/argv，都是完整105题Level3，temperature1/top_p.95/max_tokens32768、batch8或64；所有caps/errors进入105分母、>95%至少100正确。

C8先跑，因为本版实际将LM8 BF16换为W4；失败就停止候选、不开C64/benchmark。成功再C64，全部质量通过后独立CPU重核完整原始105题×2、源/控制器快照、63环境、bank/payload/native profiler峰值、Down42捕获与实际Graph联系、LM真实M8/64 W4调用及旧BF16/W8缺失。不得沿用934旧模型的质量或计数。

GPU前已明确替换C8旧BF16检查和C64旧W8检查，使用新记录和真实bank身份，不修改或伪造旧计数。Down审计使用已解决LM64过期INT8检查的完整适配，其他GU/KV/attention/scheduler/sampling和真实native admission保留。质量模型全部M1..64提前warm/probe，预热属于实际DeviceMemoryProfiler/native profile窗口，RNG不变。LM在主图外的Python调用不冒充Graph replay。无手工KV bytes，实际模型统计含全部原BF16/W8和新增W4及峰值。

最迟2026-10-23 15:58UTC停止，run_validation内部SIGALRM和进程组退出；本控制器仅Math不运行吞吐或16K。质量失败保留原始、报告，不原样重跑挑结果。代码/报告/关键结果推天数GitHub，数组/原始仅容器，无本地副本。质量完整通过前没有新合格发布或正式成绩。
