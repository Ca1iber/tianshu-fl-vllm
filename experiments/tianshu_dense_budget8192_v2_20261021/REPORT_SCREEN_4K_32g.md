# 当前整数模型预算8192：小4K提升1.00%，晋级fresh Math

本次是C64/N64的4K输入/1K输出筛选，不是原N256正式分数。候选预算8192有效均值 **5475.882663 total tok/s**，control4096为 **5421.468290**，配对 **+1.003683%**；相对此前合格版本同客户端N64参考5410.385242为 **+1.210587%**。平均TTFT **10893.926464→10766.053422ms，-1.173801%**。

两臂有效轮次spread **0.064439%/0.230687%**，后两轮均0 runtime autotune，满足提前规定的>=0.5%配对/旧参考、TTFT<=1%恶化、spread<1%门槛。六轮完整384个计时响应、全部冻结源/快照、真实进程环境、权重签名和实际调用已独立重新核验。判定值得跑fresh Math及其通过后的原N256 4K，尚未采用或更新正式分。

## 每轮结果

每臂独立启动三轮，每轮64请求，排除首轮。原serve argv13项。

| arm/budget | 轮次 | total tok/s | TTFT ms | autotune | 计入 |
|---|---:|---:|---:|---:|---|
| control4096 | 1 | 5244.332649 | 12877.163797 | 0 | 否 |
| control4096 | 2 | 5419.721517 | 10857.104510 | 0 | 是 |
| control4096 | 3 | 5423.215063 | 10930.748418 | 0 | 是 |
| candidate8192 | 1 | 4993.776983 | 12898.375040 | 4 | 否 |
| candidate8192 | 2 | 5469.566589 | 10696.993459 | 0 | 是 |
| candidate8192 | 3 | 5482.198738 | 10835.113384 | 0 | 是 |

## 原模型与真实运行条件

374项运行时源码，64项实际环境，原合格359项源码均按原SHA保留。43份原BF16/packed/scale权重签名与原合格版本完全一致。保持原GateUp M8/M64 INT8、LM小批BF16/M64INT8、BF16 prefill linear、gpu4、seq64、async、futureKV、pairedK/S16和arena8192；没有带入负收益的MLP opaque融合。

两臂warmup实际完成83/99点，最大M分别4096/8192。实际scheduler/worker配置与新外层记录一致；内层默认记录是外层修改前4096。V2独立long policy的真实worker源/哈希、实际预算/seq核验，runtime源00d748923ea041f7054d517374eb20e4064cf3ff6277353dee4ba8b6cafe71ff，归一化预算表达式后其整模块AST与原版一致。真实hook仅精确路由一次并恢复runpy，原合格源码不改。

实际KV blocks **40731→40150**，均0 preemption，所有future reservation在结束时释放。观察到候选可用KV容量减少，本轮缓存仍足以承载负载；不能写作两臂容量相同，也不据此推算16K成绩。

## 首轮四条调优的范围

99点warmup完成后，首个8192小4K仍出现四条调优：index的8192、5120 key（后者benchmark输入5089），以及GateUp linear的M8192/M5120、N12288/K2048。对应记录的耗时0.27/0.57/1.47/0.96秒；详见FIRST_ROUND_TUNING与SCREEN_ROUNDS原始行。它们没有出现在后两轮，不把99个预热点表述为冷启动后绝不会发生临时调优。

这与旧稀有shape问题有联系，但本轮已将首轮剔除且后两轮稳定；后续可单独改进这些额外路径的启动预热，不在已启动的Math冻结版本里改源码。

## 当前验证与持久化

条件队列已进入新的Math：`/workspace/benchmark-results/20261021_dense_budget_v2_math_tianshu`，105道Level3、C8、temperature1/top_p0.95/max_tokens32768，严格>95%后原N256 4K四轮。当前正式最好仍5482.79 total tok/s、Math101/105，旧正确率不借给8192。完整16K未安排。

原始筛选根目录为`/workspace/benchmark-results/20261021_dense_budget8192_v2_service_{control,candidate}_tianshu`；runner为`20261021_dense_budget8192_v2_pair_runner_tianshu`。384份完整正文、日志和源码快照只留容器；GitHub保存代码/本报告/关键JSON与CSV。V1固定4096保护的启动失败另行归档，V2两臂全部重跑。
