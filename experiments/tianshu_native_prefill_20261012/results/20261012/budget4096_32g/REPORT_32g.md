# BI-V150 32G：native Prefill预算4096确认

只改budget2048→4096，保留已验证NativeL0Prefill、Hybrid、adaptive Decode，C64/N128、4K/1K三轮排首轮。使用已完成native1控制2742.829558；候选三轮1725.843410/2721.599824/2855.799091，后两轮均值2788.699458，实测+1.672357%。TTFT7877.185953→7602.613680ms；控制调优0/0/0，候选0/1/0，一轮有新增调优，未宣称纯稳态收益。两组共782请求（各7probe+384计时）合法，greedy probe文本相同。

原确认未启动服务器：共享run_service_trial保留了误复制的旧prefill GEMM检查，强制tile1/budget2048，触发RuntimeError，父又读不存在result。已移除错误预算限制、保留tile1要求并在父先检查child exit，日志放comparison parent。修复后以上完整试验正常结束，原失败目录不覆盖。

提升未达当前明确采用门槛，不升best、不新跑Math/正式4K。默认budget2048仍为正式最佳2712.55 total tok/s、TTFT4720.38ms、Math102/105。

完整两组原始包4381710 bytes，17分块及整包SHA256 c40d73b1c04a6290f80ca952c4a285e0bdd778719d48e881c4a28bcbea47697d本地验证，保留所有响应、状态、路由、源码、日志。原失败原因日志见旧root budget4096/candidate_4096/runner.log。性能剖析TODO。未改权重、安装源码、官方benchmark和沐曦仓库，未跑优化版完整16K。
