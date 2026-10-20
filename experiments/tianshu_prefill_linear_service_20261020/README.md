# 天数真实服务：大批次 GateUp BF16 linear 路由

只读保留正式 5375.50 tok/s 的源码与权重。两个模式均使用相同的 GPU 静态打包、4 线程加载与目标表 SQL 反射，保留全部 43 个 W8A8 bank、原 BF16 权重及 M8/M64 真 INT8 路径。

启动期原生 profile、GEMM 预热均保留 qualified mm。在 ModelRunnerFL.capture_model 入口核对 42 层 M2048 预热已完成，再启用路由。control 继续 mm；candidate 对未命中 INT8 的 GateUp M>=128 使用 flag_gems.linear(x.contiguous(), weight, None)。连续化处于真实请求计算路径内。小 M 无改动。

三轮 4K/1K C64 N64，首轮不计小规模均值。记录每层、M、stride、捕获状态与 ready 后真实调用，要求 42 层在请求中实际执行。全部 61 项实际环境与完整历史注意力/调度审计保留。不是正式 N256 成绩，不声称有新 Math 结果。有收益后独立跑原始 Level3 的 105 道 Math，必须 >95% 才能接受新数学路径，再决定是否完整 4K 复测。无需例行完整 16K。
