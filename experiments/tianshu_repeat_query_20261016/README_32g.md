# 天数：保留BM16，重复单query到无效计算行

当前GQA16/2每KV head只有8个有效query行，BM16里另8行最终不写回。本候选保留BM16及原最终store mask/offset，只在受守卫的IS3D单token路径，让计算阶段另8行重复读取同一个有效query，避免其Q load/score使用原无效query掩码。有效行Q/K/V、浮点步骤、CV/KV16-w1/S8/int64/grid/reducer保留；不等同于旧BM8改变矩阵布局的失败尝试。

新constexpr REPEAT_DECODE_QUERY默认为False；只有IS3D/BM16/BQ2/Qheads16/GQA8/no TD_QO/no sinks/no QQ bias/no ALiBi/no KV量化时激活。安装wrapper只在max_seqlen_q<=1时选IS3D，0-query及额外query blocks在原kernel提前退出。其它路径是原行为。源码5处修改可精确逆回当前cv源8deb，完整epilogue及其后的reducer源码逐字相同。

CPU1152个地址/写回映射检查、原wrapper3D契约、AST与实际CLI help通过；GPU未运行。计划原生CV、生成副本False恒等控制、True候选共17场景/34候选，原与变QKV Graph/FP32/CPU抽点/尾部/真实参数/9轮交替event。micro只是筛选，不能代替服务和新Math；有明确C64收益后才进一步接入。正式4K3992.97、Math103/105不变。

运行：`python3 experiments/tianshu_repeat_query_20261016/run_probe.py --output NEW_ROOT --stop-at-utc ISO`。GPU串行且带硬截止，源码冻结10份、自动归档。
