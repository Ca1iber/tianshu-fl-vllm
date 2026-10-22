#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_native_int8_prefill_validation_20261022
screen=experiments/tianshu_native_int8_prefill_results_v2_20261022
pair=/workspace/benchmark-results/20261022_native_int8_prefill_pair_v2_runner_tianshu
runner=/workspace/benchmark-results/20261022_native_int8_prefill_validation_runner_tianshu
math8=/workspace/benchmark-results/20261022_native_int8_prefill_math8_tianshu
math64=/workspace/benchmark-results/20261022_native_int8_prefill_math64_tianshu
official=/workspace/benchmark-results/20261022_native_int8_prefill_official4k_tianshu
stop=2026-10-22T15:58:00+00:00
test ! -e "$runner";mkdir "$runner";printf '%s\n' "$$" > "$runner/runner.pid"
trap 'code=$?; if (( code!=0 )); then cp "$runner/phase.txt" "$runner/failed_phase.txt"; printf "failed\n" > "$runner/phase.txt"; printf "%s\n" "$code" > "$runner/exit_code.txt"; fi' EXIT
printf 'waiting_for_screen\n' > "$runner/phase.txt"
sha256sum "$base"/*.py "$base"/*.sh "$base/PLAN_32g.md" > "$runner/frozen_entrypoints.sha256"
while true;do
 status=$(ps -p 29810 -o stat= | tr -d ' ' || true)
 if [ -z "$status" ] || [[ "$status" == Z* ]];then break;fi
 test "$(date -u +%s)" -lt "$(date -u -d "$stop" +%s)"
 sleep 15
done
test "$(cat "$pair/phase.txt")" = complete
jq -e '.complete' "$screen/SCREEN_GATE_32g.json" >/dev/null
if ! jq -e '.candidate_passes_perf_screen' "$screen/SCREEN_GATE_32g.json" >/dev/null;then
 printf 'performance_rejected_no_Math_or_official_started\n' > "$runner/phase.txt"
 exit 0
fi
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
printf 'CPU_prepared_profiles\n' > "$runner/phase.txt"
env -u PYTHONPATH python3 -u "$base/check_prepared_profiles_cpu.py" --output /workspace/benchmark-results/20261022_native_int8_prefill_validation_CPU_tianshu > "$runner/CPU.log" 2>&1
jq -e '.complete and .CUDA_not_initialized' "$base/CPU_PREPARED_GATE_32g.json" >/dev/null
for batch in 8 64;do
 remaining=$(( $(date -u -d "$stop" +%s) - $(date -u +%s) ));test "$remaining" -gt 5400
 sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
 ixsmi | grep -q 'No running processes found'
 printf 'Math%s\n' "$batch" > "$runner/phase.txt"
 if [ "$batch" = 8 ];then root="$math8";else root="$math64";fi
 env -u PYTHONPATH python3 -u "$base/run_validation.py" --stage math --batch "$batch" --screen-root "$screen" --output "$root" --stop-at-utc "$stop" > "$runner/math${batch}.log" 2>&1
 jq -e '.phase=="complete" and .accuracy_passed and .total==105 and .correct>=100' "$root/result.json" >/dev/null
 jq -e '.complete and .total==105 and .correct>=100 and .previous_accuracy_not_reused' "$root/accuracy_audit.json" >/dev/null
done
remaining=$(( $(date -u -d "$stop" +%s) - $(date -u +%s) ));test "$remaining" -gt 2400
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
ixsmi | grep -q 'No running processes found'
printf 'official4K\n' > "$runner/phase.txt"
env -u PYTHONPATH python3 -u "$base/run_validation.py" --stage official --batch 8 --screen-root "$screen" --math-root "$math8" --math64-root "$math64" --output "$official" --stop-at-utc "$stop" > "$runner/official4k.log" 2>&1
jq -e '.phase=="complete" and .completed_runs==4' "$official/result.json" >/dev/null
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
ixsmi > "$runner/hardware_after.txt";grep -q 'No running processes found' "$runner/hardware_after.txt"
printf 'complete\n' > "$runner/phase.txt"
date -Is > "$runner/completed.txt"
