# 精确请求数 mixed graph：小规模吞吐+6.51%，未采用

2026-10-19，BI-V150，执行源码110f4ca。4K/1K、C64/N64，每边3轮排除首轮；相同原生CPU ngram栈，只有候选增加248点混合图。不补齐请求数，服务计时没有数值reference forward。

|组|轮|Total tok/s|Output tok/s|Mean TTFT ms|新增调优|
|---|---:|---:|---:|---:|---:|
|control|1|2812.354|562.471|18321.39|0|
|control|2|2943.379|588.676|14909.64|0|
|control|3|2865.962|573.192|15061.50|4|
|candidate|1|3082.287|616.457|18588.64|0|
|candidate|2|3097.723|619.545|14997.02|4|
|candidate|3|3089.984|617.997|15010.20|4|

稳态均值 2904.670→3093.853 total tok/s，+6.513068%。这说明本次框架图策略在CPU ngram栈内有收益；当前小规模结果仍低于近期非投机N64约4037 tok/s，因此不采用，不投入本候选新Math或完整N256评测。正式最好4093.81/Math102/105保持，不将不同N的数值当同期正式比较。

运行时仍出现稀有GEMM调优：控制第三轮4条，候选第二、三轮各4条。投机保护下75点prefill补齐仍关闭。不能称本次已解决运行时调优稳定性。

全部384响应、10个固定种子canary、104份冻结源码、42项实际环境、16份metrics和实际mixed replay覆盖已下载独立审计，native标准正FP32拒绝采样及原目标模型保持，未借用旧Math作本候选质量结论。

原始归档7,624,653字节，SHA-256 `af50ed7a1158c74dde51334053b4487ab79f61652bbdb138d13c4f897cb470ea`，本地原件service-results/20261019_tianshu/exact_service/20261019_exact_request_graph_service_tianshu，local_download_audit.json完整通过。

下一项独立测试原生GPU ngram，依据实际vLLM源码可保留async scheduling和async spec decode；它改变整条框架路线，不能归因于单个kernel。
