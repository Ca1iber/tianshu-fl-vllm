# 天数 BI-V150 当前正式成绩

2026-10-20，选择性W8A8成为新4K正式最佳：新Math100/105（95.2381%），完整4K4轮排首轮，1024/1024请求成功、零抢占。16K仍保留前版独立成绩，两个表项属于不同配置。

|输入/输出，C64|请求/轮|total tok/s|output tok/s|平均TTFT ms|配置|
|---|---:|---:|---:|---:|---|
|4096/1024|256|5375.50|1075.10|4191.14|Future-KV＋成对K读取＋选择性W8A8，新正式4K|
|16384/1024|128|3885.91|228.58|106814.09|前版Future-KV，非新W8A8配置16K成绩|

4K较上一版成对K5235.52提升2.6737%，较公开2028.01提升165.0628%；TTFT较上一版4161.08增加约0.72%，计分吞吐极差/均值0.2997%。新配置未完整测16K，不宣称同版完成两个正式负载。

新4K直接启动：

```bash
cd /workspace/vllm-plugin-FL
bash experiments/tianshu_dense_w8a8_release_20261020/serve_tianshu_minicpm.sh
```

如容器重建缺少生成代码，先运行同目录restore_generated_sources.sh。需保留原模型及比赛镜像依赖；FlagGems对应4cdc3b760205bb4ce7edd9099fa19fa4986adc24。入口使用Bash/jq直接exec原13项参数，核验311份运行来源、9份控制文件和57项环境；VLLM_DISABLE_COMPILE_CACHE=1避免旧AOT旁路。原BF16权重保留，额外INT8/scale1326938112bytes已计入实际profiling，真实KV40731blocks。

证据：[4K正式报告](experiments/tianshu_dense_w8a8_official_results_20261020/REPORT_32g.md)、[独立采用证明](experiments/tianshu_dense_w8a8_official_results_20261020/OFFICIAL_GATE_32g.json)、[新Math](experiments/tianshu_dense_w8a8_quality_20261020/REPORT_32g.md)、[启动说明](experiments/tianshu_dense_w8a8_release_20261020/README.md)。标签tianshu-original-command-4k-k8v16-w8a8-20261020。

上一版成对K5235.52/TTFT4161.08、Math103/105仍保留，入口experiments/tianshu_pair_release_20261020/serve_tianshu_minicpm.sh，标签tianshu-original-command-4k-k8v16-pairk-20261020。它不含W8A8。

前版同一profile完整4K5103.41、16K3885.91、Math103/105及1536/1536请求成功，入口experiments/tianshu_future_kv_release_20261020/serve_tianshu_minicpm.sh，标签tianshu-original-command-4k16k-k8v16-futurekv-20261020。

GPU静态打包、SQL反射与GateUp BF16 dispatch后续实验不在本次5375.50版本内；后续候选必须分别核验。
