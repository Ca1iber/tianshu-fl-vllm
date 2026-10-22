#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_native_int8_prefill_official_results_20261022
release=experiments/tianshu_native_int8_prefill_release_20261022
runner=/workspace/benchmark-results/20261022_native_int8_prefill_formal_analysis_runner_tianshu
formal=/workspace/benchmark-results/20261022_native_int8_prefill_official4k_v2_tianshu
pipeline=/workspace/benchmark-results/20261022_native_int8_prefill_validation_v2_runner_tianshu
stop=2026-10-22T15:58:00+00:00
test ! -e "$runner";mkdir "$runner";printf '%s\n' "$$" > "$runner/runner.pid"
printf 'waiting_for_formal\n' > "$runner/phase.txt"
trap 'code=$?; if (( code!=0 )); then cp "$runner/phase.txt" "$runner/failed_phase.txt"; printf "failed\n" > "$runner/phase.txt"; printf "%s\n" "$code" > "$runner/exit_code.txt"; fi' EXIT
sha256sum "$base/verify_final_cpu.py" "$base/PLAN_32g.md" "$base/run_after_formal.sh" "$release"/*.sh "$release/DRAFT_PLAN_32g.md" > "$runner/frozen_entrypoints.sha256"
while true;do
 status=$(ps -p 35992 -o stat= | tr -d ' ' || true)
 if [ -z "$status" ] || [[ "$status" == Z* ]];then break;fi
 test "$(date -u +%s)" -lt "$(date -u -d "$stop" +%s)"
 sleep 15
done
test "$(cat "$pipeline/phase.txt")" = complete
jq -e '.phase=="complete" and .completed_runs==4' "$formal/result.json" >/dev/null
ixsmi > "$runner/hardware_before_CPU.txt";grep -q 'No running processes found' "$runner/hardware_before_CPU.txt"
sha256sum --check --quiet /workspace/benchmark-results/20261022_native_int8_prefill_formal_analysis_frozen.sha256
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
printf 'checkpoint_and_independent_CPU\n' > "$runner/phase.txt"
sha256sum /workspace/MiniCPM5-2B/model-00000-of-00001.safetensors > "$formal/checkpoint_after.sha256"
env -u PYTHONPATH python3 -u "$base/verify_final_cpu.py" > "$runner/CPU.log" 2>&1
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
jq -e '.complete' "$base/OFFICIAL_GATE_32g.json" >/dev/null
if ! jq -e '.promotion_passed' "$base/OFFICIAL_GATE_32g.json" >/dev/null;then
 printf 'complete_no_promotion\n' > "$runner/phase.txt";exit 0
fi
printf 'qualified_release_profile\n' > "$runner/phase.txt"
jq '{old_output_root:.raw_root,old_repository_root:"/workspace/vllm-plugin-FL",working_directory:"/workspace",environment:(.profile_environment+.additional_environment),server_command,frozen_sources:.model_frozen_sources,controller_sources}' "$base/OFFICIAL_GATE_32g.json" > "$release/serve_profile_template.json"
sha256sum "$release"/*.sh "$release/serve_profile_template.json" "$release/DRAFT_PLAN_32g.md" "$base/OFFICIAL_GATE_32g.json" experiments/tianshu_native_int8_prefill_quality_results_20261022/QUALITY_GATE_32g.json experiments/tianshu_prefill_control_binary_all_20261022/QUALIFIED_MODEL_CACHE_32g.json > "$release/QUALIFIED_RELEASE.sha256"
printf 'restore_check_and_dry\n' > "$runner/phase.txt"
bash "$release/restore_frozen_cache.sh" --check > "$runner/cache_check.log" 2>&1
TIANSHU_NATIVE_INT8_PROFILE_DRY_RUN=1 TIANSHU_NATIVE_INT8_PROFILE_RECORD_DIR=/workspace/benchmark-results/native_int8_release_dry_20261022 bash "$release/serve_tianshu_minicpm.sh" > "$runner/dry.json"
jq -e --slurpfile gate "$base/OFFICIAL_GATE_32g.json" '.server_not_started and .source_validation_passed and .verified_runtime_source_count==877 and .verified_controller_source_count==($gate[0].controller_sources|length) and .command==$gate[0].server_command and .working_directory=="/workspace" and .profile_environment==.selected_actual_environment and (.profile_environment|length)==63' "$runner/dry.json" >/dev/null
jq -e --slurpfile gate "$base/OFFICIAL_GATE_32g.json" '.record_root as $root|$gate[0].raw_root as $old|($gate[0].profile_environment+$gate[0].additional_environment|with_entries(.value|=(split($old+"/")|join($root+"/"))))==.profile_environment' "$runner/dry.json" >/dev/null
cp "$runner/dry.json" "$release/DRY_RUN_GATE_32g.json"
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
ixsmi > "$runner/hardware_after.txt";grep -q 'No running processes found' "$runner/hardware_after.txt"
printf 'complete_pending_report_manifest_and_Git\n' > "$runner/phase.txt"
date -Is > "$runner/completed.txt"
