# 天数 BI-V150 当前推荐版本与已测成绩

2026-10-21：当前推荐正式4K **5586.28 total tok/s、1117.25 output tok/s、平均TTFT3780.63ms**，全新Math **102/105（97.142857%）**。N256/C64、4轮排首轮；1024/1024成功、零错误/抢占、服务就绪后无新增autotune记录。后三轮吞吐极差/均值0.13927%。

|输入/输出，C64|请求/轮|total tok/s|output tok/s|平均TTFT ms|配置范围|
|---|---:|---:|---:|---:|---|
|4096/1024|256|5586.28|1117.25|3780.63|当前prefill GateUp INT8版本，新完整4K|
|16384/1024|128|3885.91|228.58|106814.09|旧FutureKV版本，不是当前配置的16K|

4K比原默认5482.79吞吐+1.88754%、TTFT-5.84275%；比此前最高吞吐备选5509.23高1.39856%；比公开2028.01高175.45624%（2.75456倍）。历史比较不冒充同次AB；采用前独立N64配对小测+1.75127%，新Math和原N256正式均已通过。本配置没有新完整16K，不能把旧16K拼成同一profile双长度资格。

```bash
cd /workspace/vllm-plugin-FL
bash experiments/tianshu_prefill_int8_release_20261021/restore_generated_sources.sh
bash experiments/tianshu_prefill_int8_release_20261021/serve_tianshu_minicpm.sh
```

启动前确认GPU空闲。纯Bash/jq核验407运行来源、12主控制来源、62环境、原13参数后exec；VLLM_DISABLE_COMPILE_CACHE=1。26份生成源约1MB仅补缺、SHA冲突不覆盖，不含权重或编译缓存。两个额外结果CPU复核器各记SHA，不算入12主控制文件。标签 **tianshu-original-command-4k-prefill-int8-20261021**。

新GateUp在24个已预热对齐大M使用A8W8，含混合批次整矩阵；纯decode M8/M64为NN，LM8仍BF16、LM64原NT整数，其余尺寸及QKV/O/Down保留原路。原BF16权重保留，43 packed banks共1326938112bytes计入真实DeviceMemoryProfiler，新增prefill不增加payload。Math实际KV40731、正式40723，容量小差异已报告，均零抢占、FutureKV全部释放。24M预热RNG不变；实际请求新INT8尺寸和42层执行均验证。

证据：[正式4K报告](experiments/tianshu_prefill_int8_official_results_20261021/REPORT_32g.md)、[独立复核](experiments/tianshu_prefill_int8_official_results_20261021/OFFICIAL_GATE_32g.json)、[原始四轮CSV](experiments/tianshu_prefill_int8_official_results_20261021/OFFICIAL_RAW_32g.csv)、[新Math](experiments/tianshu_prefill_int8_quality_results_20261021/REPORT_32g.md)、[复现启动](experiments/tianshu_prefill_int8_release_20261021/README.md)。完整回答/raw仅容器，GitHub保留代码、报告、评分/用量/哈希和关键CSV/JSON。

## 已保留历史版本

|版本|正式4K total tok/s|TTFT ms|Math|标签/说明|
|---|---:|---:|---:|---|
|当前prefill INT8|5586.28|3780.63|102/105|tianshu-original-command-4k-prefill-int8-20261021|
|前默认BF16 prefill linear|5482.79|4015.23|101/105|tianshu-original-command-4k-lm-bf16-prefill-20261021|
|前8192预算备选|5509.23|4740.15|102/105|已被当前吞吐/TTFT同时超过|
|更早W8A8|5375.50|4191.14|100/105|tianshu-original-command-4k-k8v16-w8a8-20261020|
|更早成对K|5235.52|4161.08|103/105|tianshu-original-command-4k-k8v16-pairk-20261020|

更早同一FutureKV profile曾完成4K5103.41、16K3885.91、Math103/105，标签tianshu-original-command-4k16k-k8v16-futurekv-20261020，保留双长度旧证据。

后续 index缓存-0.2386%、DEBUG阈值-0.0748%、KV空分配+0.1209%、NN单项+0.1423%都未建立稳定模型收益；当前新prefill包含NN以共享所需物理权重，不把整体收益归给NN单项。新Down真实权重和KV fixed-K/pair-V组件仍独立探索，尚不属于本5586.28配置或正确率。

## 后续8192预算完整4K实测，保留为未建立稳定收益的候选

新8192预算prefillINT8完成原N256四轮：排首轮后三轮5582.83/5616.21/5626.15，均值**5608.40total、1121.68output tok/s、TTFT4406.24ms**；新Math**103/105（98.10%）**，1024成功、无抢占。比推荐5586.28高0.396%，但轮间极差0.772%，TTFT高16.55%，并有4条采样/索引autotune，无新增GEMM调优。没有稳定改善证据，当前推荐和复现标签不变；此数值仍是实际完整结果，不把它删去或当成N64筛选分数。详见[8192正式报告](experiments/tianshu_prefill_budget8192_official_results_20261021/REPORT_32g.md)。本候选也没有完整16K。
