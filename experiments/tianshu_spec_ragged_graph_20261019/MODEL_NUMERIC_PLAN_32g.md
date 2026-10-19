# 下一步真实模型数值门槛（待实现）

CPU42捕获元数据/2144批次组合已通过；本候选尚无GPU结果。先完成以下隔离验证再做性能，不使用旧Math为新框架背书。

1. 新hook继承已冻结tianshu_ngram_fixed_index_20261016/hooks/sitecustomize.py，外层Finder委托旧loaders后再安装ragged_policy；pin真实ModelRunner b014、Dispatcher51f、ForwardContext1c6、FL GraphWrapper ba78。class补丁须在worker初始化/显存估算/捕获之前。暂不启用旧coverage hook（其audit固定旧folder和42env）；可复用旧route_observer.py作为只读模块，但记录归本候选，必须让inspect.signature看见原始方法（policy已@wraps）。
2. 使用新独立run_model_probe/audit/verify文件，继承旧run_diagnostic的原argv、38环境基础、ngram/fixedindex/正FP32 rejection门槛，额外路由/数值模式环境如实记录。两个4096/512 C64N64诊断阶段，128响应与4canary、phase marker原子写、slot/RNG/源码证据，23:58硬停。数值观察器有同步/额外前向，仅为门槛，不报性能。
3. _model_forward回调仅在真实FULL上下文且descriptor.uniform=False、对应自有已捕获entry/实际mixed请求、maxQ2、Req16..64且全prompt完成时选样。读取CPUquery_start_loc与实际GPUcu/seq_lens/slot_mapping核对；真实请求数n可小于图容量N，实际query数q可小于T。slot[q:T]必须全-1，cu[n:N+1]必须全q，seq_lens[n:N]全0；GPU输入ids/positions实际q行留存hash/数值。
4. 8个数值样本按组选择：3个Req无补齐且T>N、3个Req有补齐且T>N、2个Req有补齐且T==N。保证变化的query pattern/实际q/序列长度，至少同一图不同输入有证据。若实际负载未覆盖所需组，保留未完成原件，不能放宽门槛后谎称成功；可追加隔离负载。
5. actual ForwardContext提供override_forward_context，可用dataclasses.replace构造eager mode的副本。先保留相同padded输入+元数据跑eager；再把ids/positions切到q，TritonAttentionMetadata替换num_actual_tokens=q和slot_mapping[:q]，ForwardContext.slot_mapping逐项切片、batch_descriptor改q/n、runtime_mode=NONE，跑真正unpadded eager。真实attention metadata是dataclass、42层可能共享同一对象，按id缓存副本保持别名，不修改原对象。
6. 原模型forward可在相同当前KV位置重复执行（输入/位置/seq_lens不前进）；但参考写入会改变当前KV值，所以必须恢复候选KV。owner.kv_caches为list tensor（可递归处理list/tuple外壳），本机dense布局预期[blocks,2,16,2,128]、42层、BF16。由真实有效slot//16选当前触及页。候选前保留未触及的守卫页，候选后clone触及页；每次eager参考后index_copy_恢复候选页，bitwise用int16视图比较以支持未初始化NaN位。守卫页须在候选/参考/恢复后保持。不能clone整份32GB KV。
7. 保存candidate hidden的clone，对比padded/unpadded eager的前q真实行，严格finite和.02+.01*abs(reference)预设门槛；再分别compute_logits做同样完整真实行比较（可先沿用此保守门槛，另记录max_abs/max_relative/bad_count及逐位状态）。必须把大GPUlogits未序列化的离线审计范围写清。投机sampling/rejection不在参考forward中重跑，forward+projection的全局GPU RNG前后state应逐位一致；返回候选clone保证后续sampling用候选结果。
8. 把guard、cache恢复、cu/slots/seqs、输入hash、hidden/logit标量差异、RNGhash写独立numeric_PID.json，错误先记原件后raise。不覆写既有Gate或测过的*.py。验证通过后生成真实NUMERIC_GATE，再同源数字模式关闭的4K/1K 3+3对照。只有明确收益再freshMath105/正式N256。

已知关键实现：自有42 mixed descriptors只在get_capture_descs加入同一FULL组，不放入native dispatcher.key set，防止(tokens==requests)键污染短prefill/Q1原路径，且避免一个mode分多组破坏native显存估算。runtime只在自己的entry已有graph时手动返回FULL，否则原路由。生成_dummy_run只改变max_query_len=2和CPUquery列表，原函数所有模型/KV/hidden gather逻辑保留；Qcapacity==Reqcapacity捕获样本含一条Q2与一条Q0。
