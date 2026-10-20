# 天数 BI-V150 当前正式版本

2026-10-20，采用未来输出KV准入组合。Math-500 Level3：103/105（98.0952%）。官方原命令完成4K和16K各4轮、各排除首轮，1536/1536请求成功、零抢占。

|输入/输出，C64|请求/轮|total tok/s|output tok/s|平均TTFT ms|
|---|---:|---:|---:|---:|
|4096/1024|256|5103.41|1020.68|4197.36|
|16384/1024|128|3885.91|228.58|106814.09|

相对上一正式版本：4K+0.0737%（基本持平），16K+4.3724%。相对公开基线2028.01/915.15分别+151.65%/+324.62%。

```bash
cd /workspace/vllm-plugin-FL
bash experiments/tianshu_future_kv_release_20261020/serve_tianshu_minicpm.sh
```

重建容器如缺少生成代码，先运行同目录restore_generated_sources.sh。原始模型及比赛镜像依赖仍需保留；该入口使用shell直接exec原始vllm命令并导出已测量配置。详细报告：[正式结果](experiments/tianshu_future_kv_official_results_20261020/REPORT_32g.md)，[独立采用证明](experiments/tianshu_future_kv_official_results_20261020/OFFICIAL_GATE_32g.json)，[启动说明](experiments/tianshu_future_kv_release_20261020/README.md)。

版本标签：tianshu-original-command-4k16k-k8v16-futurekv-20261020。元数据、RoPE/KV与CTA顺序实验均未纳入这一正式profile。
