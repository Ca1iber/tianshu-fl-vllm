#!/usr/bin/env bash
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd -- "$base/../.." && pwd -P)
results="$base/../tianshu_QKV64_keep128_official_results_20261023"
quality="$base/../tianshu_QKV64_keep128_quality_results_20261023/QUALITY_GATE_32g.json"
gate="$results/OFFICIAL_GATE_32g.json"
test "$repo" = /workspace/vllm-plugin-FL
test ! -e "$base/QUALIFIED_RELEASE.sha256"
jq -e '.complete and .promotion_passed and .official_benchmark and .source_count==928 and .original903_r3_sources_exact_subset and .environment_count==63 and .C8_correct>=100 and .C64_correct>=100 and .zero_preemptions_verified and .actual_native_Down_runtime_all42_M4096 and .M64_Down_BF16_retained and .all_inherited_runtime_and_new_Down_QKV_O_audits_recomputed' "$gate" >/dev/null
jq -e '.complete and .model_quality_qualified and .model_source_count==928 and .controller_count==51 and .QKV128_native_retained_verified and .nonuniform_M64_BF16_policy_verified' "$quality" >/dev/null
quality_sha=$(sha256sum < "$quality");quality_sha=${quality_sha%% *}
jq -e --arg h "$quality_sha" '.independent_quality_gate_sha256==$h' "$gate" >/dev/null
gzip -n -9 -c "$results/CPU_COMPARISON_32g.json" > "$results/CPU_COMPARISON_32g.json.gz"
comparison_sha=$(gzip -cd "$results/CPU_COMPARISON_32g.json.gz" | sha256sum);comparison_sha=${comparison_sha%% *}
jq -e --arg h "$comparison_sha" '.CPU_comparison_sha256==$h' "$gate" >/dev/null
gzip -cd "$results/CPU_COMPARISON_32g.json.gz" | jq -e '.profile.actual_attention_M64_capture_native_banks==42 and .profile.new_attention_packed_payload_bytes==397135872 and (.profile.actual_attention_runtime_native_by_role|all(.banks==42 and .M4096_banks==42))' >/dev/null
mkdir "$base/evidence"
evidence_tmp=$(mktemp /tmp/tianshu-QKV64-evidence.XXXXXX)
trap 'rm -f -- "$evidence_tmp"' EXIT
for batch in 8 64;do
 mathroot="/workspace/benchmark-results/20261023_QKV64_keep128_math${batch}_tianshu"
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
source "$base/write_readme.sh"
gate_sha=$(sha256sum < "$gate");gate_sha=${gate_sha%% *}
template_sha=$(sha256sum < "$base/serve_profile_template.json");template_sha=${template_sha%% *}
jq --arg gate_sha "$gate_sha" --arg template_sha "$template_sha" --slurpfile evidence "$base/CONTROLLER_EVIDENCE_32g.json" '{qualified:true,version:"tianshu-original-command-4k-uniform-QKV64-keep128-20261023-r4",formal_gate_sha256:$gate_sha,template_sha256:$template_sha,model_source_count:.source_count,controller_source_count:(.controller_sources|length),environment_count:.environment_count,command_count:(.server_command|length),total_tok_s:.total_tok_s,output_tok_s:.output_tok_s,ttft_ms:.ttft_ms,hot_spread_percent:.hot_spread_percent,C8_correct:.C8_correct,C64_correct:.C64_correct,retained_INT8_Down_payload_bytes:528826368,additional_INT8_QKV_O_payload_bytes:397135872,no_new_weight_packs_relative_r3:true,original_checkpoint_sha256:.checkpoint_postformal_sha256,original_r3_unchanged:true,full16K_not_run:true,controller_evidence_original_bytes:$evidence[0].total_original_bytes,controller_evidence_gzip_bytes:$evidence[0].total_gzip_bytes,no_GPU_started_by_packaging:true}' "$gate" > "$base/RELEASE_MANIFEST_32g.json"
sha256sum "$base/serve_tianshu_minicpm.sh" "$base/restore_frozen_cache.sh" "$base/restore_generated_sources.sh" "$base/restore_controller_evidence.sh" "$base/build_release.sh" "$base/write_readme.sh" "$base/DRAFT_PLAN_32g.md" "$base/ADDITIONAL_M64_CACHE_32g.json" "$base/QUALIFIED_MODEL_CACHE_32g.json" "$base/serve_profile_template.json" "$base/CONTROLLER_EVIDENCE_32g.json" "$base/evidence/"*.gz "$base/README_32g.md" "$base/RELEASE_MANIFEST_32g.json" "$gate" "$results/verify_cpu.py" "$results/PLAN_32g.md" "$results/CPU_COMPARISON_32g.json.gz" "$results/KEY_RUNS_32g.csv" "$quality" > "$base/QUALIFIED_RELEASE.sha256"
sha256sum --check --status "$base/QUALIFIED_RELEASE.sha256"
jq '{qualified,version,total_tok_s,output_tok_s,ttft_ms,model_source_count,controller_source_count,controller_evidence_gzip_bytes,no_GPU_started_by_packaging}' "$base/RELEASE_MANIFEST_32g.json"
