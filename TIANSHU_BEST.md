# 天数 BI-V150 当前正式成绩

2026-10-20，成对K读取成为新4K正式最佳：新Math103/105（98.0952%），完整4K4轮排首轮，1024/1024请求成功、零抢占。16K保留前版独立测得成绩，两个表项对应不同配置。

|输入/输出，C64|请求/轮|total tok/s|output tok/s|平均TTFT ms|配置|
|---|---:|---:|---:|---:|---|
|4096/1024|256|5235.52|1047.10|4161.08|Future-KV＋成对K读取，新正式4K|
|16384/1024|128|3885.91|228.58|106814.09|前版Future-KV，非新配置16K成绩|

4K相对上一版5103.41提升2.5887%，相对公开基线2028.01提升158.1605%，计分三轮极差/均值0.1129%。新配置未完整测16K，不宣称它已完成两个正式负载的完整资格验证。

4K新配置直接启动：

```bash
cd /workspace/vllm-plugin-FL
bash experiments/tianshu_pair_release_20261020/serve_tianshu_minicpm.sh
```

如容器重建导致生成代码缺失，先运行同目录restore_generated_sources.sh。原模型和比赛镜像依赖仍需保留。启动入口为Bash/jq，直接exec原始vllm参数，校验282份运行来源、8份控制文件并导出已测量54项环境。

详细证据：[4K正式报告](experiments/tianshu_pair_official_results_20261020/REPORT_32g.md)、[独立采用证明](experiments/tianshu_pair_official_results_20261020/OFFICIAL_GATE_32g.json)、[独立新Math](experiments/tianshu_pair_quality_20261020/REPORT_32g.md)、[启动说明](experiments/tianshu_pair_release_20261020/README.md)。新4K标签tianshu-original-command-4k-k8v16-pairk-20261020。

前版已同时完成4K5103.41与16K3885.91、Math103/105、1536/1536请求成功，仍可使用experiments/tianshu_future_kv_release_20261020/serve_tianshu_minicpm.sh。其证据在tianshu_future_kv_official_results_20261020，标签tianshu-original-command-4k16k-k8v16-futurekv-20261020。两版本都不包含metadata融合、uint32读取或W8A8新实验。
