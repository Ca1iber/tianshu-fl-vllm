# Down64 W4 v2：新鲜C64 103/105，独立完整CPU复核通过

冻结998971f，完整Level3 105题、并发64、temperature1/top_p.95/max_tokens32768。官方review/report：103/105=98.095238%，严格>95%通过。105请求均成功、0错误、完整分母105。104个正常stop，1个max_tokens（index83，仍按官方review计入总数并判正确）；错误题index18/87。没有丢弃cap、重跑失败题或用旧准确率代替。本结果不是对随机采样正确率提升的证明。

独立CPU再次解析105份reviews/predictions/官方report和全部raw SHA；重新执行全部旧GU/KV/attention/scheduler/sampling/LM、原Down除M64 BF16以外每条及新W4审计，完整profile与accuracy_audit逐字相同。916来源与85控制器、完整来源快照/current source SHA、63环境与13官方argv、checkpoint SHA均核对通过。

真实W4 M64捕获42 bank，原Down M64 BF16捕获0；保留原op schema/fake，通过公开register_kernel更换CUDA路由，原_init_fn/banks closure和原BF16/W8/scales指针绑定匹配；其他M委托真正原backend。42个load预热、42个真实worker memory-profile scratch探针，新增payload280756224B，实际模型参数/buffer存储计入。原QKV/O M64仍84个BF16 bank，LM64仍原W8A8；C64实际主模型M64图执行证据与W4捕获链接，不把Python次数当重放次数。

原始根 `/workspace/benchmark-results/20261023_dots_Down64_v2_math64_tianshu`。C64_QUALITY_GATE保存全部文件SHA与独立复核；C64_KEY_SCORES为105行关键结果。所有大日志/答案/数组只在容器。

C8目前正在同版本独立验证，故model_quality_qualified仍false，尚未跑本版4K或正式benchmark、未采用。正式最佳仍r3 5950.75 total tok/s。只有新鲜C8也>95%并完整复核后，才进入另行冻结短4K。
