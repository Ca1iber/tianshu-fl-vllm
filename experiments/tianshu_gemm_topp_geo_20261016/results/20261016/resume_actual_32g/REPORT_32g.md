# 几何TopP与M2048组合：全105题Math102/105，97.14%

2026-10-16 18:31完成同候选质量评测：105题全成功、零错误，102正确/105，97.142857%，满足大于95%。错题index18/87/83；104个stop与1个max_tokens（83，32768上限，不截短或加长）。该版本质量与原best3992.97的旧103/105分开记录。

用户新增六小时时段，按2026-10-17 00:00结束、23:58停止GPU任务。运行源1b7c5ddc1d156a5d6b70088cbb3b79871acb8585，58份目标运行源不变；10份额外编排及EvalScope源另存并冻结。运行中同步8f5495c仅为后续未运行CPU诊断文件，68份Math运行源同步前后全部SHA未变，用户WIP保留。

此前17:58停止时已获得该候选103条预测；容器实际EvalScope1.11.1通过ModelResult/ReviewResult解析，真实105题数据的全部question_id与原顺序匹配，CacheManager实际只留下83/101。DatasetDict通过items()访问。数据fingerprint为sha256:d2a0a846c4e8e0b074d9cf260e5bb5c3a1d6e41086e4ecefc7d17c624949151a，原新指纹相等，无不匹配强制复用。CPU原件见CPU_CACHE_AUDIT_32g.json。

新服务先以8并发64token独立探针验证实际B8几何TopP路由，保存8份完整请求响应；随后before_metrics成功数8。EvalScope用全105题数据、--use-cache和--rerun-review，仅新生成83/101，并重新评分全部105题，after_metrics成功数10、增量2。原103条预测完整文件前缀及逐行SHA不变，包括原错题和reasoning/text；没有删除错题、替换答案或导入旧正确率。

原始官方serve argv、temperature1/top_p.95/max_tokens32768/batch8、默认2048预算/256序列/75暖点与38项实际环境均核验。原生采样RNG、CV/KV16/S8、nativePrefill L0保持；候选只M2048三项GEMM tile及纯TopP32K实际max归一化与几何中点。全105原始预测/重新评分、68源快照及当前仓库源、所有请求计数和运行路由，均下载后通过本机独立verifier。

原件service-results/20261016_tianshu/gemm_topp_geo_math_resume/20261016_gemm_topp_geo_math_resume_tianshu，5439023bytes/21parts，SHA25649b4472d3c97a4c223dfa07a966605d7cdc3bde1c6ee1e96eae356e6d99c2004。所有分片和全包核验。runner49544为Z、API49546/eval50142已消失、GPU空闲后，18:33启动独立正式4K N256×4新服务：root /workspace/benchmark-results/20261016_gemm_topp_geo_official_tianshu，runner50811，原始官方serve argv、原benchmark脚本SHA不变。正式成绩尚未完成，best3992.97不变。

```bash
python3 experiments/tianshu_gemm_topp_geo_20261016/verify_validation_downloaded.py \
  --root ../service-results/20261016_tianshu/gemm_topp_geo_math_resume/20261016_gemm_topp_geo_math_resume_tianshu \
  --screen-root ../service-results/20261016_tianshu/gemm_topp_geo_service/20261016_gemm_topp_geo_service_tianshu
```

CPU调用栈/指针诊断仍未运行，待正式4K结束后独立执行。不运行16K，不另行push，也不修改沐曦仓库。
