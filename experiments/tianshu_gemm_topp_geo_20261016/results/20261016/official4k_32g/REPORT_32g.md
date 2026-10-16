# 天数原始命令4K：4093.81 tok/s，较前版提升2.53%

2026-10-16 18:57完成，原始官方serve argv和未修改benchmark脚本，4096输入/1024输出/C64/N256，连续四轮并按原脚本排除第一轮。后三轮均值4093.81 total tok/s、818.76 output tok/s、TTFT4309.35ms；比上一版3992.97提升2.525438%，TTFT下降3.028401%；比公开基线2028.01吞吐提升101.863403%。

|轮次|Total tok/s|Output tok/s|Mean TTFT ms|成功请求|是否计入均值|
|---|---:|---:|---:|---:|---|
|1|4010.66|802.13|5841.46|256/256|排除|
|2|4096.35|819.27|4304.67|256/256|计入|
|3|4095.08|819.02|4309.27|256/256|计入|
|4|4090.00|818.00|4314.10|256/256|计入|

全部1024请求成功；后三轮吞吐极差/均值0.155112%。前后metrics零抢占；服务日志未见autotuning打印（不等于证明没有任何JIT成本）。官方脚本SHA256为0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565。原脚本只保存stdout与CSV，本报告不声称取得逐请求响应正文。

本候选保留默认token预算2048、max_num_seqs256、async=True、stream_interval1、实际75个Prefill预热点、CV Decode/KV16-w1/S8/nativePrefill L0、BF16/page16/prefix cache和原采样RNG。新增M2048的O/down/gate_up GEMM tile，以及B8..64纯TopP的32K tile、实际全局max归一化和几何搜索中点；p=.95、32次上限、搜索间隔门槛与CPU严格1e-6均保持。TopK及小batch继续原路径。

同一58份目标源与38环境对应的完整Math Level3为102/105（97.142857%），105成功零错误，超过95%要求。103条原候选预测逐字节保留，只补83/101并重新评分全105题；68份目标与编排源、数据指纹、原生成参数及API计数已独立核验。参见相邻resume_actual_32g完整报告。这个正确率属于本候选，不借用前一版本103/105。

本次独立正式服务沿用官方命令，未隐藏加入额外调度参数：

```bash
export VLLM_PLUGINS=fl
vllm serve /workspace/MiniCPM5-2B --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256]]'
```

上述环境另需本实验已验证的38项算子profile。`serve_tianshu_minicpm.sh` 与 `serve_profile_template.json` 提供它们，实际启动检查58份源码SHA、清除继承的实验开关并创建独立记录目录。新脚本通过bash语法与真实shell dry-run检查，argv及38项环境映射后与实际Math API完全一致，注入的旧实验开关也会移除；未额外启动服务，仅原控制器启动的同profile服务取得本成绩。

正式运行使用源1b7c5dd的冻结目标58份，运行期间容器同步至6bac196仅改CPU诊断与报告，目标源码SHA保持。全部原始stdout/CSV、前后metrics、路由/预热记录、58份源码快照、实际38环境与对应新Math，已下载本机并独立重算通过。原件路径 `service-results/20261016_tianshu/gemm_topp_geo_official/20261016_gemm_topp_geo_official_tianshu`，4247850bytes/17分片，SHA256 `1d9b2d74b4bd678489e805b66c93bc0057bf19f2cd3b98c14f882e1ff188bda0`；质量归档SHA `49b4472d3c97a4c223dfa07a966605d7cdc3bde1c6ee1e96eae356e6d99c2004`。

```bash
python3 experiments/tianshu_gemm_topp_geo_20261016/verify_validation_downloaded.py \
  --root ../service-results/20261016_tianshu/gemm_topp_geo_official/20261016_gemm_topp_geo_official_tianshu \
  --math-root ../service-results/20261016_tianshu/gemm_topp_geo_math_resume/20261016_gemm_topp_geo_math_resume_tianshu \
  --screen-root ../service-results/20261016_tianshu/gemm_topp_geo_service/20261016_gemm_topp_geo_service_tianshu
```

采用范围为已验证的原始serve命令4K工作负载；该版未测16K，也不据此宣称全比赛提交资格。旧3992.97与历史自选调度参数4091.29各自保留。正式服务和控制器终态、GPU空闲后，另行启动CPU调用诊断探索框架开销；诊断不会被当成新吞吐成绩。用户此前指定的GitHub检查点已推，本次后续工作先保留本机/容器提交，不另行push。
