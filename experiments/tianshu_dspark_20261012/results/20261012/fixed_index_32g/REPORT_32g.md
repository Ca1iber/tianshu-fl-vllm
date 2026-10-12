# BI-V150 32G：投机长输出索引失败与双路径修复

当前正式最好仍为 2712.55 total tok/s、542.51 output tok/s、TTFT 4720.38 ms，独立 Math 102/105。此报告不产生新正式成绩。

匹配输入种子的 4K/1K、C64/N128 三轮控制：1564.914107、2677.333696、2785.502471 total tok/s，排首轮均值 2731.418083，384 请求成功，三轮运行期调优均为零。控制来自已验证的 native Prefill+Hybrid+adaptive，budget2048，原始模型。

选中的双 attention tile DSpark 在首次 1K 输出确认中失败：没有完成性能轮次，记录 113 次调优，日志报告耗时合计约848秒。旧错误字段为空，无法据此认定超时或 OOM。4K/128 短输出的 8133.276910 不能代表正式 4K/1K 成绩。

固定索引第一版同样未完成首轮。新错误追踪结合服务端日志确认：FlagGems `index` 相邻维度路径 `_linearized_adjacent_index_func` 在请求期进行自动调优，随后 `get_empty_cache_for_benchmark` 申请256 MiB，GPU只剩156 MiB，触发 `torch.OutOfMemoryError`，引擎退出。客户端 `AssertionError` 是流式返回不完整的后果。该轮完成16条索引调优记录。

第一版只替换 `_index_func`，遗漏了相邻维度优化分支，实际固定索引路由记录为0。这是修复覆盖错误，不能据此宣称固定launch方案没有作用。

修复同时包装两条内部调用路径。只在轴0、一个1D整数索引、1D/2D源、BF16/F32/I32/I64且连续输出的已验证范围使用固定Triton row-gather。保留公共API的原有类型、形状和广播处理，其余路径回退。支持合法负索引和源/索引stride；契约只覆盖合法索引，越界错误语义未声明兼容。

验证：原row-gather 56项CPU逐位一致性通过；双路径修复通过48项真实FlagGems公共API检查，包括连续/跨步源、I32/I64索引、四种源dtype、宽1/2048/130560，确认每次命中固定路径。另有3次CUDA Graph重放逐位检查，在捕获后修改源内容及索引，宽257检验列尾部。

双路径修复的4K/1K三轮重跑已启动。截至18:41 UTC+8，首轮进行中，实际固定索引路由3类，请求期索引调优记录0；尚无完成轮次、性能结论或独立Math成绩。后续按已明确的收益、正确率门槛进行质量与正式4K验证。

未修改模型权重、安装目录、原始benchmark脚本或沐曦仓库。全部实验在天数私有仓库的实验目录内，通过显式进程开关启用。原始benchmark SHA256为0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565。

原始失败日志、源代码快照、数值结果和所有已产生的响应已分块SHA256校验并备份到本地service-results/20261012_dspark_tianshu；具体清单见同目录backup_manifest_*.json。失败归档wrapper的complete仅代表证据封存完成，experiment_succeeded为false，未重构不存在的响应。

性能剖析：TODO，未采集固定索引版的profiler。此次只有故障日志与功能验证。正式性能仍沿用保留标签tianshu-best-native-prefill-20261012。
