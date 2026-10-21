# 2026-10-21 中午：三项已闭环，推荐版不变

预约至10-22 00:00，GPU硬停23:58北京。推荐仍407源tianshu-original-command-4k-prefill-int8-20261021，正式4K5586.28/TTFT3780.63ms，Math102/105。

TopP16K/grid2服务小测+0.83956%，独立新Math99/105=94.2857%，未达>95%，不采用、不重跑刷分；正式4K未启动。完整105评分/usage/hash已经归档tianshu_topp_grid_current_quality_results_20261021，438源/62env/13控制器检查通过；原runner224721已Z终态，API/evalscope退出，不可按旧STATE以为仍运行。

Down INT8 SplitK v1在首次tail编译失败，保留原目录失败；v2仅host计算CHUNK后传constexpr，36配置/108三阶段大矩阵全GPU数值、独立全部134498304 A量化、选中INT64/每段partial，以及54小尾部完整CPU输出通过。4K fastest1.157647ms，比同次BF16 1.103415和原unsplit INT8 1.039579都慢；不接模型、不做新Math。v2 runner227232和CPU229127均已Z，runner目录phase=complete。报告在tianshu_int8_down_splitk_v2_20261021。

TopP只改变grid（保持原GEO32K/16K、w16/s1）全部300组/600完整输出、33实际Sampler mask/token/RNG逐位相同；CPU229729已完成重建2.8796B FP32值/22056行，changed_rows0。关键B64 BF16组件变化约±0.5%，无明确收益，不做服务/Math。新结果在tianshu_topp_grid32_bitwise_api_results_20261021；GPU229129已Z，当前GPU空闲。

下一步先查现有attention integer-PV原语：tianshu_k8vi8_pv_int8_20261021原语已GPU/CPU通过，但没有完整真实writer+attention性能。README和PRIMITIVE_GPU_REPORT给出后续范围。此前单纯V8 BF16消费者比当前K8V16慢39%，不能重复当新方法。另只读研究真正INT8 QK是否值得，不可把已有反量化BF16 dot称作INT8 dot。

实际模型config：LlamaForCausalLM、hidden2048、16 Qheads/2 KVheads、head_dim128，无qk_norm配置。不要误以为Q32/KV4/HD64或假设QK有RMSNorm硬界。真实reader已每KV共享8个Q heads，双KV-head打包也已测负收益，避免重复。

代码/报告/关键结果仅天数容器+GitHub，禁止本地副本，不改用户两份dirty gateup_silu源码，FlagGems保持clean4cdc3b7，沐曦只读。GitHub分支exp/tianshu-v-int8-20261020。SSH master/tmp/tianshu-20261021-control，16:27前刷新。没有新完整16K，服务benchmark每轮及时报告。
