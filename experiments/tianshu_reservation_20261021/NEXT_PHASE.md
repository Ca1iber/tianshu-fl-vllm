# 续约后的下一阶段

使用窗口至北京时间2026-10-22 00:00；新任务stop-at-utc=2026-10-21T15:58:00+00:00。所有之前GPU/CPU队列均已终止，无旧的全局午夜守护。数据只留天数容器与天数GitHub，沐曦仓库只读。

当前正式4K5375.50 total/1075.10 output、TTFT4191.14ms、新Math100/105、1024/1024零抢占，版本dcb7a6c与标签tianshu-original-command-4k-k8v16-w8a8-20261020已推送。新profile无完整16K；旧3885.91属于future-KV profile。

1. 先改善后续试验启动效率。实际W8装载打包133.413秒；组件统一4线程CPU12.491秒/GPU8.489秒不是实际worker线程配置的严格对照。新独立策略分别记录默认CPU、CPU4、GPU4的实际Torch线程数、43组payload SHA、真实FL DeviceMemoryProfiler、packing/ready时间和C8短请求。SQL目标表反射已在1158表43784行完成全结构/值核验；实际导入的site-packages sql.py与仓库副本SHA相同，接入应针对真实方法。不要修改已经验证版本的冻结来源；新文件与新profile记录独立来源和完整实际环境。

2. 再做prefill真实服务对照。连续M2048/M4096的原生linear组件约比mm快20%，row-stride问题经真实Graph内contiguous修正后4case全部通过。使用已确认的ModelRunnerFL.capture_model入口，在原始profile/warmup之后安装候选路由；记录内层原生MM预热发生在外层路由之前，不能把旧MM记录当作新实际prefill路径。现有W8整数M8/M64保持，两臂冷加载策略一致。候选真实GateUp prefills可考虑M>=128走linear(x.contiguous(),W,None)，其他路径须有明确回退。审计实际42个prefix的新prefill执行，尤其Math必须实际走到新分支。完整选择环境逐项核验，不能通过过滤新键伪造旧计数。先4K C64 N64屏测，明显收益后新105题Math严格>95%、再原N256完整4K；不每次跑完整16K。

3. 后续探索统一4K/16K的显存与批次收益。W8当前保留原BF16 GPU权重导致额外1.326GB、KV blocks40731。可以独立组件测试扩展GUP/LM的M32/40/48/56整数配置，以及原BF16参数保留CPU、GPU只留量化表示的可能性；如需prefill BF16临时反量化，其完整成本必须包含在计时，不能重复把已失败的软件W8 decode当作新收益。所有参数/缓冲区、Torch compile真实参数依赖、实际显存profiling、全权重哈希和新质量必须验证。只有显存/调度策略发生实质改变并有收益证据时才安排16K小屏测和必要完整测，绝不合并不同profile为同版成绩。
