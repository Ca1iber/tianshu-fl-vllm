#!/usr/bin/env bash
# Sourced by the qualified builder after all frozen quality/performance checks.
{
 cat <<'DOC'
# 天数 BI-V150：均匀整批QKV64原生INT8，保留QKV128，r4

在已发布r3原903项模型输入的精确字节基础上，共928项输入冻结。仅对FULL/64tokens/64requests/均匀解码/无LoRA的QKV64增加原生INT8，其余QKV64和所有O64保留BF16；QKV128及原512/4096路径继续原生INT8。没有合入LM925 pad候选。原MiniCPM5-2B checkpoint和模型结构不变，无微调。

DOC
 jq -r '"正式4K（4096/1024/C64/N256，四轮排除第一轮）：**\(.total_tok_s) total tok/s、\(.output_tok_s) output tok/s，TTFT \(.ttft_ms)ms**。热三轮极差/均值\(.hot_spread_percent)%。相对r3历史正式5950.75提高\(.vs_previous_formal_percent)%；同期同请求N64对照提高\(.short_same_run_gain_percent)%。历史正式比较存在时段差异，短测不是正式N256成绩。\n\n新鲜完整Math-500 Level3：C8 **\(.C8_correct)/105**、C64 **\(.C64_correct)/105**，均严格>95%，所有达到输出上限的题目计入分母。原始逐条评审独立重算、实际42层M4096路径、QKV64原生42层捕获、O64 BF16、全部继承审计、1024请求成功、零抢占、就绪后无Triton autotune均通过。Python路由计数不冒充图重放次数。\n"' "$gate"
 cat <<'DOC'
QKV/O仍只有原84份NN打包副本，共397135872 bytes；保留原BF16权重，全部计入实际DeviceMemoryProfiler，KV预算不手填。新增M64严格kernel缓存8个文件与原384项缓存共392项，只上传SHA清单；模型、依赖、缓存、题目、预测全文和数组均在容器。没有新16K结果，也不据此声称完整双项比赛资格已审核。

```bash
cd /workspace/vllm-plugin-FL
bash experiments/tianshu_QKV64_keep128_release_20261023/restore_generated_sources.sh
bash experiments/tianshu_QKV64_keep128_release_20261023/restore_frozen_cache.sh --check
bash experiments/tianshu_QKV64_keep128_release_20261023/restore_controller_evidence.sh --check
TIANSHU_NATIVE_INT8_PROFILE_DRY_RUN=1 bash experiments/tianshu_QKV64_keep128_release_20261023/serve_tianshu_minicpm.sh
bash experiments/tianshu_QKV64_keep128_release_20261023/serve_tianshu_minicpm.sh
```

六份必要控制JSON缺失时，无参数运行restore_controller_evidence.sh可从随版gzip恢复精确字节；已存在的不同SHA文件拒绝覆盖。启动器核验全部模型/控制器输入，清理继承实验环境，保留原13项serve参数与63项环境，仅重定向记录路径。QUALIFIED_RELEASE.sha256记录发布工具、正式门槛、独立核验源码、关键CSV、压缩比较和质量证据。
DOC
} > "$base/README_32g.md"
