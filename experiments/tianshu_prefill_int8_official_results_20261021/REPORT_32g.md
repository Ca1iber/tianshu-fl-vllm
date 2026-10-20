# 天数正式4K：5586.28 tok/s，Math102/105，采用新版prefill INT8

2026-10-21 07:02北京时间完成。原始vllm serve 13项参数、未改官方脚本SHA `0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565`；4096输入/1024输出、C64、N256，连续4轮排除第1轮。**后三轮5586.28 total tok/s、1117.25 output tok/s、平均TTFT3780.63ms**。比原推荐5482.79吞吐+1.88754%、TTFT-5.84275%；比此前最高吞吐备选5509.23高1.39856%；比公开2028.01高175.45624%（2.75456倍）。这些历史比较不冒充同次AB；本版先有独立N64配对小测+1.75127%，后有本次原N256正式验证。

|轮|total tok/s|output tok/s|Mean TTFT ms|成功|计分|
|---|---:|---:|---:|---:|---|
|1|5420.72|1084.14|5676.02|256/256|排除|
|2|5583.62|1116.72|3834.25|256/256|是|
|3|5583.81|1116.76|3753.58|256/256|是|
|4|5591.40|1118.28|3754.05|256/256|是|

共1024/1024成功、零错误，后三轮吞吐极差/均值0.139270%。独立Decimal重算原CSV均值与四轮stdout一致；API Application startup complete之后覆盖全部四轮，新增 `Triton autotuning for function` 记录为0。零抢占、FutureKV预留全部释放。没有据此声称不存在任何其它JIT成本。

## 改动及质量

仅对GateUp预热过的24个对齐大M使用A8W8，包括混合批次的整块矩阵；其它尺寸保留原BF16回退。GateUp M8/M64使用真实NN物理布局，LM M8保持BF16、M64保持原NT INT8；QKV/O/Down等没有被本轮改成INT8。没有新packed权重payload，原BF16权重/检查点均保留，43 packed banks的1326938112bytes在真实DeviceMemoryProfiler内计入。NN单独小测此前+0.1423%未建立收益，本版包含NN是新prefill所用同一物理权重布局，不能把本轮收益归给NN单项。

新Math Level3 **102/105=97.142857%**，105成功、0错误、incomplete=false，104stop+1max_tokens（索引83仍正确，全部105计分），355190输出tokens。严格>95%，没有复用旧正确率或重跑求过线。temperature1/top_p.95/max_tokens32768、batch8、timeout3600按官方要求。逐题评分、usage、行SHA独立重算，完整回答只容器。Math实际新prefill M256覆盖42层，官方真实新prefill M128/2048/3328/3840/4096覆盖42层；C8 Math不声称测到了C64全部精度路径。

小测、Math、正式均精确407份运行来源、62项环境、原13参数；Math/正式额外相同12份主控制来源。24个M预热的A量化/选定真实W CPU INT64→BF16核验和CPU/GPU RNG不变通过；真实请求42层NN layout、INT8 kernel/config/binary及LM分支均有记录。Math物理KV40731 blocks，正式40723 blocks，实际启动有8块差异，未隐瞒容量变化；额外payload字节一致、均无抢占。JSON中Python调用次数是保存快照下界，不能当Graph replay总数。

采用规则预先保存于PLAN_32g.md：相对原默认吞吐至少+1%、TTFT不高于原1.01倍、吞吐波动<1%、公开4K性能门及新Math通过、就绪后无autotune。本版全部通过，独立OFFICIAL_GATE adopted=true；原始comparison保留不覆盖。结果目录两个额外CPU复核器各记自身SHA，不冒充原冻结12主控制文件。

## 复现与范围

`../tianshu_prefill_int8_release_20261021/serve_tianshu_minicpm.sh` 直接Bash/jq核验407/12/62后exec原始命令；26份生成源约1MB仅补缺，不含权重/编译缓存。当前默认5586.28替换5482.79。旧8192预算5509.23/TTFT4740.15不再是最高吞吐。

本配置没有完整16K；3885.91属于旧FutureKV配置，不能合并声明同一profile双长度合格。完整原始JSON/日志/硬件/metrics仅容器 `/workspace/benchmark-results/20261021_prefill_int8_official4k_tianshu`，GitHub保留原CSV/summary、比较摘要与原件SHA、关键审计、报告和代码。可见GPU的CPU导入检查只发生在排除首轮，发现初始化CUDA上下文后，后续CPU准备和独立审计使用CUDA_VISIBLE_DEVICES=-1；没有与计分轮并行的第二个GPU计算任务。

后续真实Down INT8和KV fixed-K/pair-V组件不属于本版成绩；它们以独立目录和唯一GPU队列继续探索，不自动改本推荐配置。

GitHub的COMPARISON_SUMMARY只删去约4.3MB完整profile执行跟踪；ARCHIVE记录容器原comparison的完整SHA/字节数，其它比较字段原值保留。原始完整comparison未改写。发布前已校验Git内两份CSV与容器原CSV逐字节相同。
