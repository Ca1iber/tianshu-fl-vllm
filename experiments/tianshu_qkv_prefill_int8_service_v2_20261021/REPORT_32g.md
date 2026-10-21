# QKV prefill INT8：实际42层生效，N64仅+.293%，未达预声明门槛

2026-10-21，BI-V15032G，原13serve参数、62env、432来源绑定；现行407源全部保持，新增QKV专属辅助/模型策略。每臂4K输入/1K输出、C64/N64三轮，排首轮。384/384计分请求成功，源快照/完整usage/seed-canary、C8与C64 Graph、现有43bank、FutureKV及零抢占检查保持，独立verify_pair完成，六轮请求期间autotune均0。

|臂/轮|total tok/s|TTFT ms|
|---|---:|---:|
|control1，排除|5270.2544|12549.2698|
|control2|5480.5182|10373.5154|
|control3|5482.1041|10415.1541|
|candidate1，排除|5286.9839|12382.2768|
|candidate2|5495.1699|10244.5875|
|candidate3|5499.5928|10263.8673|

后两轮均值5481.311164→5497.381391tok/s，+.293182%；TTFT10394.334729→10254.227422ms，−1.347920%。两臂极差.028933%/.080455%，虽稳定、TTFT改善，但吞吐未达预声明>.5%，不采用、不做新Math/N256。推荐仍正式5586.28/TTFT3780.63/Math102。不能把这次N64结果当正式N256成绩。

候选真实42层QKV进入新prefill算术；原方法委托的fallback与decode记录保持。24个原生capture尺寸各自完整A量化和选中真实W INT64输出检查通过，运行观察M128/2048/4096的5586次Python调用，另42次fallback；计数不等于Graph replay或模型总调用。静态完整42层220200960个权重的NumPy独立量化、checkpoint全部hash和layer0旧组件hash一致。新增packed+scale220631040B，原BF16保留，实际尚未退出的原生DeviceMemoryProfiler内创建。

native model_memory_usage control6429220352、candidate6650899968B；新packed总payload1326938112→1547569152B，实际KV40133→39586块。不能按payload单一因素解释全部KV块差。源验证和真实原生预算未手工改写。RUNTIME_BINDING保留全部42bank、profiler来源、warmups和实际路由/二进制hash，原始PT/完整回答/日志仅容器。

control和candidate均增加opaque custom-op，再从内部委托原apply；该证据不能证明与原407编译上下文或BF16内核逐位相同。故候选出分前已预声明独立未包装407直接对照（tianshu_qkv_prefill_exact_reference_20261021，92974e7）。本轮未过原>.5%门槛，额外参考GPU、Math和正式均未启动，不能单凭包装后control宣称超过现行推荐。

V1曾因通用routing.py遮蔽框架模块在加载模型前失败；原失败/源保持。V2仅给辅助模块专属名并新增native routing解析检查，完整bank manifest逐字节相同；数值/性能门槛未放宽。原GPU runner232213已Z、candidate API233355/core233513退出，GPU空闲。原件根/workspace/benchmark-results/20261021_qkv_prefill_int8_service_v2_{control,candidate}_tianshu。
