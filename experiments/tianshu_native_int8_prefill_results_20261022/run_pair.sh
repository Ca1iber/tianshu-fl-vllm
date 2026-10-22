#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_native_int8_prefill_results_20261022
service=experiments/tianshu_native_int8_prefill_service_v2_20261022
runner=/workspace/benchmark-results/20261022_native_int8_prefill_pair_runner_tianshu
smoke=/workspace/benchmark-results/20261022_native_int8_prefill_service_v2_smoke_candidate_tianshu
stop=2026-10-22T15:58:00+00:00
test ! -e "$runner"
mkdir "$runner"
printf '%s\n' "$$" > "$runner/runner.pid"
trap 'code=$?; if (( code!=0 )); then cp "$runner/phase.txt" "$runner/failed_phase.txt"; printf "failed\n" > "$runner/phase.txt"; printf "%s\n" "$code" > "$runner/exit_code.txt"; fi' EXIT
sha256sum "$base"/*.py "$base"/*.sh "$base/PLAN_32g.md" "$service/CPU_GATE_32g.json" > "$runner/frozen_entrypoints.sha256"
printf 'waiting_for_smoke\n' > "$runner/phase.txt"
while true;do
 status=$(ps -p 20183 -o stat= | tr -d ' ' || true)
 if [ -z "$status" ] || [[ "$status" == Z* ]];then break;fi
 test "$(date -u +%s)" -lt "$(date -u -d "$stop" +%s)"
 sleep 10
done
jq -e '.phase=="complete" and (.screen_rounds|length)==1' "$smoke/result.json" >/dev/null
for arm in control candidate;do
 sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
 ixsmi | grep -q 'No running processes found'
 printf '%s\n' "$arm" > "$runner/phase.txt"
 env -u PYTHONPATH python3 -u "$service/run_service_trial.py" --mode "$arm" --output "/workspace/benchmark-results/20261022_native_int8_prefill_pair_${arm}_tianshu" --stop-at-utc "$stop" > "$runner/$arm.log" 2>&1
 jq -e '.phase=="complete"' "/workspace/benchmark-results/20261022_native_int8_prefill_pair_${arm}_tianshu/result.json" >/dev/null
done
printf 'GPU_pair_complete_pending_CPU_comparison\n' > "$runner/phase.txt"
sha256sum --check --quiet "$runner/frozen_entrypoints.sha256"
env -u PYTHONPATH python3 -u "$base/verify_pair_cpu.py" > "$runner/CPU_comparison.log" 2>&1
ixsmi > "$runner/hardware_after.txt"
grep -q 'No running processes found' "$runner/hardware_after.txt"
printf 'complete\n' > "$runner/phase.txt"
date -Is > "$runner/completed.txt"
