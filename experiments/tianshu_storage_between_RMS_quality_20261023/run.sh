#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_storage_between_RMS_quality_20261023
sha256sum --check --status "$base/EXECUTION.sha256"
runner=/workspace/benchmark-results/20261023_storage_between_RMS_quality_runner_tianshu
mkdir "$runner"
tr '\0' ' ' < /proc/18102/cmdline | grep -F 'tianshu_RMS_A8_GU_quality_v2_20261023/run_quality.sh' > "$runner/outer_command.txt"
kill -STOP 18102
trap 'kill -CONT 18102 2>/dev/null || true' EXIT
while [ -e /proc/18883/stat ] && [ "$(awk '{print $3}' /proc/18883/stat)" != Z ];do
 test "$(date -u +%s)" -lt "$(date -u -d '2026-10-23 15:58:00 UTC' +%s)"
 sleep 10
done
ixsmi > "$runner/hardware_before.txt"
grep -q 'No running processes found' "$runner/hardware_before.txt"
state=/workspace/benchmark-results/20261023_RMS_A8_GU_v2_math64_tianshu/state.json
jq -r '(.frozen_sources+.controller_sources)|to_entries[]|.value+"  "+.key' "$state" > "$runner/PROTECTED_INPUTS.sha256"
jq -r '.files|to_entries[]|.value+"  "+.key' experiments/tianshu_prefill_control_binary_all_20261022/QUALIFIED_MODEL_CACHE_32g.json >> "$runner/PROTECTED_INPUTS.sha256"
sha256sum --check --status "$runner/PROTECTED_INPUTS.sha256"
CUDA_VISIBLE_DEVICES='' python3 "$base/deduplicate_immutable_snapshots_cpu.py" > "$runner/dedup.log" 2>&1
sha256sum --check --status "$runner/PROTECTED_INPUTS.sha256"
ixsmi > "$runner/hardware_after.txt"
grep -q 'No running processes found' "$runner/hardware_after.txt"
jq --argjson count "$(wc -l < "$runner/PROTECTED_INPUTS.sha256")" '.+{protected_model_controller_and384cache_input_count:$count,protected_input_SHA_before_after:true,C64_running_process_not_suspended:true,only_outer_quality_shell_paused_between_stages:true,no_GPU_timing_modified:true}' /workspace/benchmark-results/20261023_RMS_quality_snapshot_dedup_CPU_tianshu/GATE_32g.json > "$base/GATE_32g.json"
kill -CONT 18102
trap - EXIT
printf '%s\n' 'SNAPSHOT_DEDUP_PASSED_OUTER_QUALITY_RESUMED'
