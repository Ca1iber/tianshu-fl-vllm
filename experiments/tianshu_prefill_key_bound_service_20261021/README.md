# KV prefill长度收紧的独立模型AB，CPU准备完毕、尚未用卡

以已发布407运行源/62环境/4096调度预算prefillINT8版本为基线，仅候选在native Route上附加CPU精确prefill fullkey bound，并在唯一dense prefill实例复制kwargs缩小maxK。保留原Route/GPUcu、decode全局kwargs、KV存储/写入/参数/采样/8192pages arena。Component24场景1.55亿有效元素和全部无效区域/保护位通过；独立policy12项实际native Route AST通过。候选的CPU元数据处理开销真实计入，control不计算未使用的bound。

新模型profile额外钩子在所有既有Worker初始化后接入，严格vendor/model/BF16/nonquant/non-spec/non-LoRA/singleparallel和原4096/64/async调度配置、pipeline/dense实际身份、元数据契约/源码SHA。捕获后标active。计时stage不增加文件写入，复用现有dense.records observer；ready阶段原/收紧pages、decode数量、实际调用来源保存。原Graph C8/C64、42层/43bank、预热和FutureKV等审计保持。保存的Python次数不是Graph replay。

N64C64 4K/1K每臂3轮排首轮、同原13serveargv；同seed两次cache_salt canary每次重算，跨臂文本作为附加检查；不能代替Math。必须显示actual_effect_exercised才可把吞吐差归到这个改动。没有触发tightening时如实报告，不伪造覆盖。源门禁425/62，CPU_GATE绑定两臂完整源SHA，GPU开始后不修改或添加.py/.sh；真实代码失败用新目录，纯元数据失败用独立CPU审计恢复。

run_queue.sh不会自动运行；要求当前staticcompile队列complete和GPU空闲，硬停止2026-10-21T15:58UTC。不能与static或其他GPU实验并行。若小测有足够稳定收益，再新Math105题严格>95%，再原N256正式4K。没有当前KV模型Math/吞吐结果，没有arena4096或完整16K结果，不据此改推荐版本。将容量缩小是后续独立因子。

CPU准备阶段补充原生内存可观测性：两臂在原determine_available_memory返回后记录weights、activation peak、non-torch、CUDA Graph estimate和实际返回KVbytes。保持原返回/配置，不增加profile_run、不重置峰值，只在启动时沿用binding文件落盘。用于之后区分暂时编译峰值/非torch/Graph估计，不据此提高KV预算。门禁最终sourcecount以新CPU_GATE为准。
