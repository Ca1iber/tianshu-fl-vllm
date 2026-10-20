# 采样预热服务已真实执行，热缓存对照未确认吞吐收益

原serve命令、4K/1K、C64/N64，每臂三轮剔首轮，384/384响应usage复核通过。两臂均为原359源合格模型精度、4096预算、seq64/async；增加启动观测后冻结371源、62实际环境，模型GateUp整数、小批次LM BF16/64整数与预填充linear保持。运行时确认原capture返回后才执行候选192次私有生成器预热，默认CPU/GPU RNG保持，seeded canary跨两臂一致。

|组别/轮次|total tok/s|mean TTFT ms|新增autotune|
|---|---:|---:|---:|
|control1，剔除|5228.13|12975.88|0|
|control2|5403.81|10891.44|0|
|control3|5430.37|10882.46|0|
|candidate1，剔除|5229.50|13006.40|0|
|candidate2|5414.51|10911.22|0|
|candidate3|5429.82|10916.31|0|

排首轮后控制5417.090189、候选5422.165916 tok/s，+0.093698%，不足以确认稳定吞吐收益。TTFT10886.948369→10913.762126ms；吞吐极差/均值0.490393%→0.282231%。两臂均零后续调参，候选实际预热0.048032s；该测试没有人为清空cache，不能用它证明冷态收益。实际KV40731/40723，零抢占，Future-KV释放通过。没有新Math或N256完整4K，也未采用。

随后源码分析发现，原生LibEntry在do_not_specialize参数上按整数类型缓存，N值变化本身不构成新的入口key；offset跨2^31时进入i64路径才会重新调参，B37恰好为跨界时真实批次。192次调用不能当成64个独立调参key覆盖的证据。四类入口受控冷缓存实验另在[精简预热实验](../tianshu_sampler_exp_cold_v2_20261021/README.md)记录，不修改本次已冻结代码与原始成绩。

原始根：`/workspace/benchmark-results/20261021_sampler_exp_service_{control,candidate}_tianshu`。关键SCREEN_GATE/SCREEN_ROUNDS/KEY_ROUNDS和源码在GitHub，回答、完整日志与tensor只留容器。
