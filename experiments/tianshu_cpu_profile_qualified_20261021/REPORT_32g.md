# 当前合格版CPU调用诊断完成

2026-10-21，使用5482.79推荐版本的359份模型父来源，364份初始诊断来源、66项实际环境，之后另外冻结观察到的调用栈源码。原13项serve参数和模型策略不变。128个4K/256输出C64响应、8个短C8 smoke、2个seed777 canary的实际usage与文本复现、43份权重签名和C64/C8真实绑定均通过；没有新模型精度或正式吞吐成绩。

|16帧观测调用|host wall中位数 ms|thread CPU中位数 ms|
|---|---:|---:|
|execute_model|7.301394|7.302886|
|sample_tokens|3.480208|3.480809|
|scheduler schedule|3.194093|3.195238|

cProfile有侵入性，不是未插桩成本。GPU异步C调用可以含等待；父子累计时间重叠；scheduler和worker在不同进程/窗口，不逐帧对应，不能将三行相加当服务耗时或据此推断GPU空转。

execute_model中_prepare_inputs累计均值5.121ms，_prepare_input_ids累计0.853ms。索引候选配置每帧重建17个Config，实际parent为index._index_candidate_configs→runtime.get_tuned_config→TunedConfigLoader.get_tuned_config，deepcopy累计约0.496ms/帧。另见每帧13次FlagGems DEBUG日志记录构造，来自add/alias/copy/fill/index/sub/to，即使once过滤重复位置，LogRecord仍会先构造。上述是诊断线索，未证明去掉它们会改善完整吞吐。

scheduler每帧64次KV分配：allocate_slots累计均值1.499ms，含FutureKV外层包装后累计更高；不能将这些重叠数字相加。现有融合元数据组件曾改善局部但服务持平，因此下一步优先做小规模实际A/B，不将微测倍数作为成绩。

DIAGNOSTIC_GATE给出每阶段按local均值排序的前30项；SUMMARY给出中位数。原始10MB worker与1MB scheduler JSON、完整响应和源码快照仅留容器`/workspace/benchmark-results/20261021_cpu_profile_qualified_tianshu`，SHA见RAW_SHA256SUMS。调用栈源码与整66项环境已复查。运行器complete，退出GPU已空闲，后续paired-K tile组件串行开始。
