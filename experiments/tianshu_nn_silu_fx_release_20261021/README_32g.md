# 5618.01正式4K配置部署

在已配置天数BI-V150比赛容器的/workspace/vllm-plugin-FL执行，指定一个不存在的结果目录：

```bash
python3 experiments/tianshu_nn_silu_fx_release_20261021/serve.py \
  --output /workspace/benchmark-results/nn_silu_fx_deployment
```

入口只调用已正式验证的candidate_profile，严格校验450源、62环境、同一13项原serve命令与两项质量门。--prepare-only只生成部署元数据，不启动GPU。正式5618.01 total/1123.60 output tok/s、TTFT3835.46ms；质量101/105及C64补充103/105。这个入口没有自动启动评测或新实验。

具体结果及原CPU审核字段恢复的说明见../tianshu_nn_silu_fx_official_results_20261021/REPORT_32g.md。旧5586.28版本TTFT略好（3780.63ms），本版按总吞吐记录新高。
