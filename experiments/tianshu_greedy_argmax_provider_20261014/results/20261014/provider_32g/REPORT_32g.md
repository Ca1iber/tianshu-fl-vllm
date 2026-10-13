# 天数 BI-V150 32g：Argmax提供者核对

明确区分原生Torch、显式flag_gems.argmax和自定义两阶段实现。实际模型CUDA注册来自FlagGems，独立进程注册则为Torch CUDA。30场景覆盖B1/B7/B64及stride/V13、ties/NaNs/Inf/大数，候选30/30精确、输入位与RNG不变；显式FlagGems有NaN首位置反例，全部数值通过为false，不能把该诊断当成全提供者正确性通过。此反例未改安装源。

|形状|Torch ms|FlagGems ms|候选 ms|
|---|---:|---:|---:|
|B1/V130560/strided=False|0.047103|0.051911|0.012144|
|B7/V130560/strided=False|0.148698|0.052605|0.014303|
|B64/V130560/strided=False|0.560078|0.086043|0.063106|
|B64/V130560/strided=True|0.570635|0.202929|0.074233|
|B3/V13/strided=True|0.011918|0.011856|0.011958|

实际B64连续输入候选相对FlagGems每步只省约23微秒，不是相对Torch的497微秒。该量级与真实服务仅+0.029499%的结果一致，不作为独立因果证明。没有新增服务benchmark或Math；正式最好3640.16不变。

NaN反例：5个，详见audit_32g.json。硬件profile：TODO有效图内kernel覆盖。

原包2577629bytes/10分片，SHA256 8b073eb52805bf0d923e6bbdfa8fed503b66d4d62786449f7782eccc87a51927，分片/整包/source_probe已本地核验。原件service-results/20261014_tianshu/greedy_argmax_provider/20261014_greedy_argmax_provider_tianshu。
