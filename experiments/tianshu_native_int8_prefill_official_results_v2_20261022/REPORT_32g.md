# 原生INT8预填充：正式4K新推荐版

官方原脚本4K/1K、C64/N256四轮排首轮：**5777.22 total / 1155.45 output tok/s，平均TTFT 3459.44 ms**。

|版本|总tok/s|输出tok/s|平均TTFT ms|
|---|---:|---:|---:|
|旧正式|5695.95|1139.19|3663.77|
|本版|5777.22|1155.45|3459.44|

相对旧正式吞吐 **+1.43%**，TTFT **-5.58%**。这项是历史比较；近期同源N64 A/B提升 **2.16%**，不能混用N64与N256成绩。相对公开2028.01总tok/s约 **2.849倍**。

热三轮差幅 **0.6353%**；1024请求全成功，零抢占，就绪后Triton autotune日志记录0。新16K未复测，不引用旧16K作为本版成绩。

## 四轮原始值（首轮排除）

|轮次|总tok/s|输出tok/s|平均TTFT ms|成功请求|
|---|---:|---:|---:|---:|
|1（排除）|5612.8|1122.56|5167.75|256.0|
|2|5759.14|1151.83|3469.91|256.0|
|3|5776.69|1155.34|3455.07|256.0|
|4|5795.84|1159.17|3453.35|256.0|

## 实现与质量

复用已有W8 NN bank和相同A8量化，仅在M512/4096使用原生torch._int_mm、INT32中间值、原BF16舍入与SiLU。原M8/M64、LM、其他M和KV路径保持已有实现，没有改模型结构、训练、微调或改写checkpoint，没有新增另一份静态权重。192MiB临时INT32工作区在原生profile内实际执行全42层并计入KV预算，没有手调KV大小或旁路常驻缓冲。

全新官方Math500 Level3 C8与真实C64各105题，均101/105=96.1905%，210请求全成功。C8预填充实际全BF16、INT8命中0，其原GPU audit_failed保留，独立CPU按真实范围复核，不把未命中的路径当作精度验证。C64原GPU完整审计complete，实际M4096原生INT8全42层、M64融合Graph重放与LM64 INT8通过，105题自然stop无截断。

同877模型源、原471精确子集、原13项serve argv，62项继承环境加显式TRITON_CACHE_DIR共63项实际观察。C64质量仅将FUTURE_KV准入开关改control以实现真实并发，native allocator参数/返回不变，差异完整记录。

## 正式程序原失败与独立恢复范围

原benchmark四轮全部SUCCESS，但原收尾审核为audit_failed，KeyError('workload')；原result/CSV均未改。缺失字段只用于long_runtime_audit的4K零抢占条件。新独立CPU从冻结原命令[[4096,1024,64,256]]和全部4轮CSV/token数唯一恢复workload审核上下文，继续执行全部原检查，包括zero_preemptions=true，再重算后三轮均值与差幅。OFFICIAL_GATE明确保留original_experiment_passed=false和CPU_workload_metadata_reconciliation=true，不声称原GPU程序审核通过。

完整模型/控制器快照、原23尺寸完整binary、新native kernel binary、全部43权重bank、attention/KV/scheduler/sampling、原M8/M64和LM、RNG/临时workspace检查均复核。原checkpoint最终SHA仍为14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c。

CPU_COMPARISON包含完整运行时profile，仅留容器并加入.gitignore；GitHub上传精简OFFICIAL_GATE、KEY_RUNS、源码和本报告。原始数据在/workspace/benchmark-results/20261022_native_int8_prefill_official4k_v2_tianshu。

直接Bash启动入口和无GPU dry证据见../tianshu_native_int8_prefill_release_v2_20261022/README_32g.md。旧5695.95发布版和全部失败记录保留。
