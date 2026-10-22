#!/usr/bin/env bash
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd -- "$base/../.." && pwd -P)
results="$base/../tianshu_LM_pad64_official_results_20261022"
gate="$results/OFFICIAL_GATE_32g.json"
quality="$base/../tianshu_LM_pad64_quality_results_v2_20261022/QUALITY_GATE_32g.json"
recovery="$base/../tianshu_LM_pad64_audit_recovery_20261022/RECOVERY_GATE_32g.json"
test "$repo" = /workspace/vllm-plugin-FL
test ! -e "$base/QUALIFIED_RELEASE.sha256"
jq -e --slurpfile q "$quality" '.complete and .promotion_passed and .official_benchmark and .source_count==925 and .environment_count==63 and .C8_correct/105>0.95 and .C64_correct/105>0.95 and (.C64_original_experiment_passed==false) and .C64_CPU_runtime_audit_recovered and .zero_preemptions_verified and .all_inherited_runtime_and_new_Down_QKV_O_audits_recomputed and .actual_native_Down_runtime_all42_M4096 and .M64_Down_BF16_retained and .model_frozen_sources==$q[0].model_frozen_sources and $q[0].model_quality_qualified' "$gate" >/dev/null
jq -e '.complete and (.original_experiment_passed==false) and .correct==103 and .new_actual_Math_guard_execution_traced and .all_other_runtime_checks_recomputed and (.recovery_controller_sources|length)==5' "$recovery" >/dev/null
gzip -n -9 -c "$results/CPU_COMPARISON_32g.json" > "$results/CPU_COMPARISON_32g.json.gz"
comparison_sha=$(gzip -cd "$results/CPU_COMPARISON_32g.json.gz" | sha256sum);comparison_sha=${comparison_sha%% *}
jq -e --arg h "$comparison_sha" '.CPU_comparison_sha256==$h' "$gate" >/dev/null
gzip -cd "$results/CPU_COMPARISON_32g.json.gz" | jq -e '.profile.actual_attention_M64_capture_native_banks==0 and .profile.new_attention_packed_payload_bytes==397135872 and (.profile.actual_attention_runtime_native_by_role|all(.banks==42 and .M4096_banks==42)) and (.profile.new_LM_actual_runtime_M|index(64)!=null)' >/dev/null
mkdir "$base/evidence"
evidence_tmp=$(mktemp /tmp/tianshu-LM-evidence.XXXXXX)
trap 'rm -f -- "$evidence_tmp"' EXIT
for batch in 8 64;do
 mathroot=/workspace/benchmark-results/20261022_LM_pad64_math8_tianshu
 if [ "$batch" -eq 64 ];then mathroot=/workspace/benchmark-results/20261022_LM_pad64_math64_CPU_recovery_tianshu;fi
 for name in result.json accuracy_audit.json controller_sources.json;do
  original="$mathroot/$name"
  expected=$(jq -er --arg p "$original" '.controller_sources[$p]|select(type=="string")' "$gate")
  actual=$(sha256sum < "$original");actual=${actual%% *};test "$actual" = "$expected"
  relative="evidence/math${batch}_${name}.gz"
  gzip -n -9 -c -- "$original" > "$base/$relative"
  packed_sha=$(sha256sum < "$base/$relative");packed_sha=${packed_sha%% *}
  actual=$(gzip -cd -- "$base/$relative" | sha256sum);test "${actual%% *}" = "$expected"
  jq -n --arg original "$original" --arg relative "$relative" --arg expected "$expected" --arg packed_sha "$packed_sha" --argjson raw_bytes "$(stat -c %s "$original")" --argjson packed_bytes "$(stat -c %s "$base/$relative")" '{original_path:$original,gzip_file:$relative,sha256:$expected,gzip_sha256:$packed_sha,original_bytes:$raw_bytes,gzip_bytes:$packed_bytes}' >> "$evidence_tmp"
 done
done
original=/workspace/benchmark-results/20261022_LM_pad64_math64_tianshu/result.json
expected=$(jq -er '.original_result_sha256' "$recovery")
actual=$(sha256sum < "$original");test "${actual%% *}" = "$expected"
jq -e '.phase=="audit_failed"' "$original" >/dev/null
relative=evidence/math64_ORIGINAL_audit_failed_result.json.gz
gzip -n -9 -c "$original" > "$base/$relative"
packed_sha=$(sha256sum < "$base/$relative");packed_sha=${packed_sha%% *}
jq -n --arg original "$original" --arg relative "$relative" --arg expected "$expected" --arg packed_sha "$packed_sha" --argjson raw_bytes "$(stat -c %s "$original")" --argjson packed_bytes "$(stat -c %s "$base/$relative")" '{original_path:$original,gzip_file:$relative,sha256:$expected,gzip_sha256:$packed_sha,original_bytes:$raw_bytes,gzip_bytes:$packed_bytes,original_failure_preserved:true}' >> "$evidence_tmp"
jq -s '{complete:true,exact_raw_bytes_preserved:true,six_quality_controller_JSON_plus_original_C64_failed_result:true,no_prompts_predictions_or_arrays:true,entries:.,total_original_bytes:(map(.original_bytes)|add),total_gzip_bytes:(map(.gzip_bytes)|add)}' "$evidence_tmp" > "$base/CONTROLLER_EVIDENCE_32g.json"
cache=/workspace/benchmark-results/20261022_prefill_control_binary_all_GPU_tianshu/qualified_model_cache
sha256sum --check --status "$base/../tianshu_LM_pad64_quality_20261022/LM_COMPONENT_CACHE.sha256"
tar -C "$cache" -czf "$base/evidence/LM_component_cache.tar.gz" CC465OJPQC6NIUG7OQ4S4K656JNKIXFO2UQZL3LBSDJTCMJGJ7HA WH2KTRV74VEZKGIBK2W6AVJIRAGC7YPOI6KF42A2JNBMPMXPUOHQ
jq '{working_directory:"/workspace",old_repository_root:"/workspace/vllm-plugin-FL",old_output_root:.raw_root,server_command:.server_command,environment:(.profile_environment+.additional_environment),frozen_sources:.model_frozen_sources,controller_sources:.controller_sources}' "$gate" > "$base/serve_profile_template.json"
source "$base/write_readme.sh"
gate_sha=$(sha256sum < "$gate");gate_sha=${gate_sha%% *}
template_sha=$(sha256sum < "$base/serve_profile_template.json");template_sha=${template_sha%% *}
jq --arg gate_sha "$gate_sha" --arg template_sha "$template_sha" --slurpfile evidence "$base/CONTROLLER_EVIDENCE_32g.json" '{qualified:true,version:"tianshu-original-command-4k-LM-pad64-20261022-r4",formal_gate_sha256:$gate_sha,template_sha256:$template_sha,model_source_count:.source_count,controller_source_count:(.controller_sources|length),environment_count:.environment_count,command_count:(.server_command|length),total_tok_s:.total_tok_s,output_tok_s:.output_tok_s,ttft_ms:.ttft_ms,hot_spread_percent:.hot_spread_percent,C8_correct:.C8_correct,C64_correct:.C64_correct,C64_original_experiment_passed:false,C64_CPU_runtime_audit_recovered:true,original_failed_C64_evidence_packaged:true,original_checkpoint_sha256:.checkpoint_postformal_sha256,original_r3_unchanged:true,full16K_not_run:true,controller_evidence_original_bytes:$evidence[0].total_original_bytes,controller_evidence_gzip_bytes:$evidence[0].total_gzip_bytes,additional_LM_weight_copies:0,LM_component_cache_groups:2,no_GPU_started_by_packaging:true}' "$gate" > "$base/RELEASE_MANIFEST_32g.json"
sha256sum "$base/serve_tianshu_minicpm.sh" "$base/restore_frozen_cache.sh" "$base/restore_generated_sources.sh" "$base/restore_controller_evidence.sh" "$base/build_release.sh" "$base/write_readme.sh" "$base/DRAFT_PLAN_32g.md" "$base/serve_profile_template.json" "$base/CONTROLLER_EVIDENCE_32g.json" "$base/evidence/"*.gz "$base/README_32g.md" "$base/RELEASE_MANIFEST_32g.json" "$gate" "$results/verify_cpu.py" "$results/PLAN_32g.md" "$results/CPU_COMPARISON_32g.json.gz" "$results/KEY_RUNS_32g.csv" "$quality" "$recovery" "$base/../tianshu_LM_pad64_quality_20261022/LM_COMPONENT_CACHE.sha256" "$base/../tianshu_native_int8_attention_prefill_release_20261022/QUALIFIED_RELEASE.sha256" > "$base/QUALIFIED_RELEASE.sha256"
sha256sum --check --status "$base/QUALIFIED_RELEASE.sha256"
jq '{qualified,version,total_tok_s,output_tok_s,ttft_ms,model_source_count,controller_source_count,controller_evidence_gzip_bytes,no_GPU_started_by_packaging}' "$base/RELEASE_MANIFEST_32g.json"
