#!/usr/bin/env bash
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd -- "$base/../.." && pwd -P)
results="$base/../tianshu_native_int8_down_prefill_official_results_v2_20261022"
gate="$results/OFFICIAL_GATE_32g.json"
test "$repo" = /workspace/vllm-plugin-FL
test ! -e "$base/QUALIFIED_RELEASE.sha256"
jq -e '.complete and .promotion_passed and .official_benchmark and .source_count==890 and .environment_count==63 and .C8_correct==102 and .C64_correct==103 and .zero_preemptions_verified and .actual_native_Down_runtime_all42_M4096 and .M64_Down_BF16_retained' "$gate" >/dev/null
gzip -n -9 -c "$results/CPU_COMPARISON_32g.json" > "$results/CPU_COMPARISON_32g.json.gz"
comparison_sha=$(gzip -cd "$results/CPU_COMPARISON_32g.json.gz" | sha256sum);comparison_sha=${comparison_sha%% *}
jq -e --arg h "$comparison_sha" '.CPU_comparison_sha256==$h' "$gate" >/dev/null
mkdir "$base/evidence"
evidence_tmp=$(mktemp /tmp/tianshu-down-evidence.XXXXXX)
trap 'rm -f -- "$evidence_tmp"' EXIT
for batch in 8 64;do
 mathroot="/workspace/benchmark-results/20261022_native_int8_down_prefill_math${batch}_v2_retry1_tianshu"
 for name in result.json accuracy_audit.json controller_sources.json;do
  original="$mathroot/$name"
  expected=$(jq -er --arg p "$original" '.controller_sources[$p]|select(type=="string")' "$gate")
  actual=$(sha256sum < "$original");actual=${actual%% *};test "$actual" = "$expected"
  relative="evidence/math${batch}_${name}.gz"
  gzip -n -9 -c -- "$original" > "$base/$relative"
  actual=$(gzip -cd -- "$base/$relative" | sha256sum);actual=${actual%% *};test "$actual" = "$expected"
  compressed_sha=$(sha256sum < "$base/$relative");compressed_sha=${compressed_sha%% *}
  raw_bytes=$(stat -c %s "$original");packed_bytes=$(stat -c %s "$base/$relative")
  jq -n --arg original "$original" --arg relative "$relative" --arg expected "$expected" --arg packed_sha "$compressed_sha" --argjson raw_bytes "$raw_bytes" --argjson packed_bytes "$packed_bytes" '{original_path:$original,gzip_file:$relative,sha256:$expected,gzip_sha256:$packed_sha,original_bytes:$raw_bytes,gzip_bytes:$packed_bytes}' >> "$evidence_tmp"
 done
done
jq -s '{complete:true,exact_raw_bytes_preserved:true,only_six_key_controller_JSON:true,no_prompts_predictions_or_arrays:true,entries:.,total_original_bytes:(map(.original_bytes)|add),total_gzip_bytes:(map(.gzip_bytes)|add)}' "$evidence_tmp" > "$base/CONTROLLER_EVIDENCE_32g.json"
jq '{working_directory:"/workspace",old_repository_root:"/workspace/vllm-plugin-FL",old_output_root:.raw_root,server_command:.server_command,environment:(.profile_environment+.additional_environment),frozen_sources:.model_frozen_sources,controller_sources:.controller_sources}' "$gate" > "$base/serve_profile_template.json"
mapfile -t scores < <(jq -r '[.total_tok_s,.output_tok_s,.ttft_ms,.hot_spread_percent,.vs_previous_formal_percent,.vs_previous_TTFT_percent,.short_same_run_gain_percent,.source_count,(.controller_sources|length)]|.[]' "$gate")
{
 cat <<'EOF'
# 天数 BI-V150：Down prefill INT8，r2

保持原始 MiniCPM5-2B 文件和结构，只在 M128/512/4096 为 Down 使用 native A8W8。M64 和其他尺寸保留原 BF16 路径；已发布 r1 的全部 877 个文件保持原字节。新版本独立保存，没有覆盖旧 r1。

EOF
 printf '正式 4K（4096/1024/C64/N256，4 轮剔除第 1 轮）：**%.2f total tok/s**、**%.2f output tok/s**，平均 TTFT **%.2f ms**，后三轮最大差距/均值 **%.4f%%**。相对上一版正式 5777.22 total tok/s 提高 %.3f%%，TTFT 相对 3459.44 ms 变化 %.3f%%。\n\n' "${scores[0]}" "${scores[1]}" "${scores[2]}" "${scores[3]}" "${scores[4]}" "${scores[5]}"
 printf '历史正式比较有时段差异；本次另做了相同输入、精确 r1 基准的 N64 短对照，提升 **%.4f%%**，候选热轮波动 0.0724%%。\n\n' "${scores[6]}"
 cat <<'EOF'
新鲜完整 Math-500 Level 3：C8 **102/105（97.14%）**，C64 **103/105（98.10%）**，原始逐条评审已独立重算。C8 本轮未命中 INT8 Down；C64 已验证 M128/4096 全 42 层真实执行，M64 图仍 BF16。达到输出上限的请求也全部计分。零抢占、无服务就绪后的 Triton autotune、所有源文件/控制器快照和继承的运行检查通过独立核验。

额外 INT8 Down 权重和缩放约 504.33 MiB，原 BF16 权重保留。对照短测的实际 KV 容量从 642784 降至 621440 tokens，分配与预热纳入模型内存评估。没有重跑 16K，不能把此结果当作完整双项比赛成绩。

在天数容器中启动：

```bash
cd /workspace/vllm-plugin-FL
bash experiments/tianshu_native_int8_down_prefill_release_v2_20261022/restore_generated_sources.sh
bash experiments/tianshu_native_int8_down_prefill_release_v2_20261022/restore_frozen_cache.sh --check
bash experiments/tianshu_native_int8_down_prefill_release_v2_20261022/restore_controller_evidence.sh --check
TIANSHU_NATIVE_INT8_PROFILE_DRY_RUN=1 bash experiments/tianshu_native_int8_down_prefill_release_v2_20261022/serve_tianshu_minicpm.sh
bash experiments/tianshu_native_int8_down_prefill_release_v2_20261022/serve_tianshu_minicpm.sh
```

若六份控制证据 JSON 缺失，可运行 `restore_controller_evidence.sh` 无参数，仅从随版 gzip 恢复缺失的精确字节，已存在但哈希不符时拒绝覆盖。这些是冻结配置和路由审计，没有上传题目、预测全文、数组或日志。实际模型文件、依赖环境及编译缓存留在容器；运行前逐项验证。所有正式 benchmark 参数保持原始 13 参数，恢复 63 项环境，仅重定向记录输出路径并清理继承的实验变量，不启动额外参数配置。

EOF
 printf '发布配置核验 %s 个模型输入和 %s 个控制器输入，启动前必须全部哈希一致。完整关键成绩见相邻 `tianshu_native_int8_down_prefill_official_results_v2_20261022`，质量见 `tianshu_native_int8_down_prefill_quality_results_v2_20261022`，短测见 `tianshu_native_int8_down_short4k_v2_20261022`。\n' "${scores[7]}" "${scores[8]}"
} > "$base/README_32g.md"
gate_sha=$(sha256sum < "$gate");gate_sha=${gate_sha%% *}
template_sha=$(sha256sum < "$base/serve_profile_template.json");template_sha=${template_sha%% *}
jq --arg gate_sha "$gate_sha" --arg template_sha "$template_sha" --slurpfile evidence "$base/CONTROLLER_EVIDENCE_32g.json" '{qualified:true,version:"tianshu-original-command-4k-native-int8-down-prefill-20261022-r2",formal_gate_sha256:$gate_sha,template_sha256:$template_sha,model_source_count:.source_count,controller_source_count:(.controller_sources|length),environment_count:.environment_count,command_count:(.server_command|length),total_tok_s:.total_tok_s,output_tok_s:.output_tok_s,ttft_ms:.ttft_ms,hot_spread_percent:.hot_spread_percent,C8_correct:.C8_correct,C64_correct:.C64_correct,additional_INT8_Down_payload_bytes:528826368,original_checkpoint_sha256:.checkpoint_postformal_sha256,original_r1_unchanged:true,full16K_not_run:true,controller_evidence_original_bytes:$evidence[0].total_original_bytes,controller_evidence_gzip_bytes:$evidence[0].total_gzip_bytes,no_GPU_started_by_packaging:true}' "$gate" > "$base/RELEASE_MANIFEST_32g.json"
sha256sum "$base/serve_tianshu_minicpm.sh" "$base/restore_frozen_cache.sh" "$base/restore_generated_sources.sh" "$base/restore_controller_evidence.sh" "$base/build_release.sh" "$base/DRAFT_PLAN_32g.md" "$base/serve_profile_template.json" "$base/CONTROLLER_EVIDENCE_32g.json" "$base/evidence/"*.gz "$base/README_32g.md" "$base/RELEASE_MANIFEST_32g.json" "$gate" "$results/verify_cpu.py" "$results/PLAN_32g.md" "$results/CPU_COMPARISON_32g.json.gz" "$results/KEY_RUNS_32g.csv" "$base/../tianshu_native_int8_down_prefill_quality_results_v2_20261022/QUALITY_GATE_32g.json" > "$base/QUALIFIED_RELEASE.sha256"
sha256sum --check --status "$base/QUALIFIED_RELEASE.sha256"
jq '{qualified,version,total_tok_s,output_tok_s,ttft_ms,model_source_count,controller_source_count,controller_evidence_gzip_bytes,no_GPU_started_by_packaging}' "$base/RELEASE_MANIFEST_32g.json"
