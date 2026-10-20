# 天数 BI-V150 当前正式成绩

2026-10-21：新4K最好 **5482.79 total tok/s**，新Math **101/105（96.1905%）**。官方每轮N256/C64、4轮排首轮；1024/1024成功，零抢占。16K仍是前版独立成绩，配置不同。

|输入/输出，C64|请求/轮|total tok/s|output tok/s|平均TTFT ms|配置|
|---|---:|---:|---:|---:|---|
|4096/1024|256|5482.79|1096.56|4015.23|GateUp W8A8、LM小批次BF16/64整数、BF16预填充linear、新4K|
|16384/1024|128|3885.91|228.58|106814.09|前版Future-KV，非本版16K成绩|

4K比上一版5375.50提升1.9959%，比公开2028.01提升170.3532%；TTFT改善约4.20%，后三轮极差/均值0.20665%。本版没有完整16K成绩。

```bash
cd /workspace/vllm-plugin-FL
bash experiments/tianshu_lm_precision_release_20261021/serve_tianshu_minicpm.sh
```

如容器重建缺少生成代码，先运行同目录restore_generated_sources.sh。启动使用Bash/jq直接exec原13项参数，核验359运行来源、10控制文件与62环境，VLLM_DISABLE_COMPILE_CACHE=1。原BF16权重保留，额外43 bank共1326938112bytes已计入真实profiling，KV40731blocks。生成源包26份.py+1关键JSON约1MB，权重和raw不上传。

证据：[完整4K](experiments/tianshu_lm_precision_official_results_20261021/REPORT_32g.md)、[独立复核](experiments/tianshu_lm_precision_official_results_20261021/OFFICIAL_GATE_32g.json)、[新Math](experiments/tianshu_lm_precision_quality_20261021/REPORT_32g.md)、[启动说明](experiments/tianshu_lm_precision_release_20261021/README.md)。标签tianshu-original-command-4k-lm-bf16-prefill-20261021。

前版W8A8 5375.50、TTFT4191.14、Math100/105，入口experiments/tianshu_dense_w8a8_release_20261020/serve_tianshu_minicpm.sh，标签tianshu-original-command-4k-k8v16-w8a8-20261020。

前版成对K5235.52、Math103/105，入口experiments/tianshu_pair_release_20261020/serve_tianshu_minicpm.sh，标签tianshu-original-command-4k-k8v16-pairk-20261020。

前版同一profile完整4K5103.41、16K3885.91、Math103/105，入口experiments/tianshu_future_kv_release_20261020/serve_tianshu_minicpm.sh，标签tianshu-original-command-4k16k-k8v16-futurekv-20261020。

后续批次补齐和K8+V整数量化组件不在5482.79配置内；候选需要独立正确性、实际执行路径及服务性能验证。
