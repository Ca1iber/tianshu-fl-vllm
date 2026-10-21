# 缓存状态确实改变本样例结果：无observer的原版本受控对照

2026-10-21 14:50完成。原407份源码、62环境项、13项serve argv，未安装stage观察器，未改变任何模型算法或采样参数。8个短请求全部完成，CPU独立复核通过，原代码冻结hash通过，GPU已空闲。

## 结果

所有请求都是同一128-token输入、seed777、temperature1、top_p.95、16个输出token。A/B/C三个cache_salt只用于隔离前缀，首先另测无salt的一对请求。接口return_token_ids用于观察，未请求logprobs。

|组|prefix queries增量|prefix hits增量|生成token序列|
|---|---:|---:|---|
|unsalted first|128|0|cold序列|
|unsalted repeat|128|112|cached序列|
|A cold / cached|各128|0 / 112|cold / cached|
|B cold / cached|各128|0 / 112|cold / cached|
|C cold / cached|各128|0 / 112|cold / cached|

4份冷缓存序列完全相同；4份热缓存序列也完全相同；两组之间第13、14个生成token不同：cold为137、248，cached为120、277，其余14个token相同。这对应之前stage canary中的一个可见字符差异，不能把字符数当token差异数。

独立CACHE_CANARY_GATE_32g.json保存8份完整token序列、实测prefix计数和比较结果。冷/冷相同、热/热相同、冷/热不同，无salt首次/重复也不同。原API返回的usage含prompt_tokens_details:null；V2核对三个必需计数并允许附加元数据，原始完整响应先保存再验证。

## 原因边界和对后续实验的影响

在没有observer的正式原版本中，控制缓存状态即可稳定复现本输入的变化。因此同种子重复prompt的冷热缓存差异足以使之前的canary文本相等检查失败，不能直接据此认定stage观察器改变模型数值。

路由原件另投影到ROUTE_EVIDENCE_32g.json：42层M128的prefill_A8W8各4次，共168调用；42层M16的delegate_qualified也各4次，共168调用。命中112个前缀token后只需计算16个prompt token；当前策略只对已预热且M>=128的形状走prefill INT8，M16回退原路径。它符合批次/缓存改变计算路径的解释。本实验没有分别剥离GEMM、attention等各算子贡献，不能仅凭路由记录断言全部差异只由INT8造成，也不能推广成所有输入均会变化。

后续新实验的前后canary应使用不同的隔离cache_salt，并记录双方实际prefix hits均为0，再比较相同seed/input的结果。正式benchmark与Math命令保持；本次没有新的正确率或吞吐成绩，没有修改模型推荐。

上一stage实验仍保留failed和缺失after_metrics/完整runtime_audit的事实；其40帧只作部分诊断，不补造缺失证据或改成通过。前向88.56%、采样9.38%的stream区间可帮助选方向，但不是单kernel活跃时间或新性能分数。

## 运行与保存

代码准备5fec90c已推送。runner239191终态Z，服务239204/core239219已退出，零preemption。raw根`/workspace/benchmark-results/20261021_prefill_cache_canary_v2_tianshu`，result SHA256 `bedfa215e611a3272d77f108b64c096b80485a7dc4b81456fa513a883fd26fa5`。原407源码与正式RUNTIME_BINDING_GATE逐项相同，实际进程62环境和原argv已读取核对。

V1因完整usage字典比较拒绝，原失败保留；V2不重跑stage/GPU性能或Math，只取得此前未完成的缓存因果对照。所有完整响应、日志、源码快照只在容器，GitHub保存代码、报告、小型gate/路由投影，本地无副本。

正式推荐仍521c154/tag tianshu-original-command-4k-prefill-int8-20261021，4K总5586.28/输出1117.25 tok/s、TTFT3780.63ms、Math102/105。
