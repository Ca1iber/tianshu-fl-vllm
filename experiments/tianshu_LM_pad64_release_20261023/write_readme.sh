#!/usr/bin/env bash
# Sourced by builder after independent formal and quality gates.
{
cat <<'TEXT'
# 天数 BI-V150：LM Head M32..64补齐融合，r4

原MiniCPM5-2B checkpoint与结构不变，无训练微调。仅在已发布r3上将LM Head实际M32..64量化输入补至64，保留BF16舍入并融合输出FP32转换；M<=8以及M9..31继续BF16。旧QKV/O、Down/GU、KV和框架策略保留，原903源逐字节包含在新925源中，无额外LM权重副本。新kernel与其内存探针位于真实DeviceMemoryProfiler内，未手调KV预算。

TEXT
jq -r '"正式4K（4096/1024/C64/N256，4轮排除首轮）：**\(.total_tok_s) total tok/s、\(.output_tok_s) output tok/s，TTFT \(.ttft_ms) ms**；热轮极差/均值\(.hot_spread_percent)%。相对历史r3正式5950.75的增幅\(.vs_previous_formal_percent)%，历史比较存在时段差异。同轮N64短对照增幅\(.short_same_run_gain_percent)%，短测不能作为正式成绩。\n\n新完整Math-500 Level3 C8=\(.C8_correct)/105，C64=\(.C64_correct)/105，均严格>95%；全部105题计入，错误与生成上限题不剔除。"' "$gate"
cat <<'TEXT'

C64原始实验完成全部题生成后，旧LM计数审计报audit_failed，原失败结果保持不变。独立CPU恢复先复现旧错误，仅更正实际新LM执行计数来源，保留原阈值与其他所有审计，再对同一105份结果独立重计与完整运行时核验；没有重复生成或重新抽样。五个CPU恢复输入与原82个GPU控制输入分开记录。原C64失败result与恢复证据均随版保留，不能称原完整验证器通过。

全部1024正式请求成功、零抢占、服务就绪后无Triton autotune；原13serve参数/63环境项保持，源和控制器逐项SHA核验。新增补齐kernel两组16文件缓存有独立恢复脚本；旧r3的生成源和384文件严格缓存继续校验。完整日志、数组、题目预测与模型只在容器，GitHub只代码、报告、关键结果和必要小型证据压缩包。未跑新16K，不代表完整比赛双项成绩或全部资格已审核。

```bash
cd /workspace/vllm-plugin-FL
bash experiments/tianshu_LM_pad64_release_20261023/restore_generated_sources.sh
bash experiments/tianshu_LM_pad64_release_20261023/restore_frozen_cache.sh --check
bash experiments/tianshu_LM_pad64_release_20261023/restore_controller_evidence.sh --check
TIANSHU_NATIVE_INT8_PROFILE_DRY_RUN=1 bash experiments/tianshu_LM_pad64_release_20261023/serve_tianshu_minicpm.sh
bash experiments/tianshu_LM_pad64_release_20261023/serve_tianshu_minicpm.sh
```

--check只核验；去掉--check可从随版证据恢复缺失文件，已有文件哈希不一致时拒绝覆盖。依赖模型、旧r3严格缓存等仍须存在于天数环境。发布脚本不启动GPU；正式成绩、质量、原审计失败与恢复来源分别见相邻official_results、quality_results_v2和audit_recovery目录。
TEXT
} > "$base/README_32g.md"
