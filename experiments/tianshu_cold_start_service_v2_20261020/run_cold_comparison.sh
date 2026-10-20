#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_cold_start_service_v2_20261020
runner=/workspace/benchmark-results/20261020_cold_start_v2_comparison_runner_tianshu
ixsmi | grep -q 'No running processes found'
test ! -e "$runner";mkdir "$runner"
printf '%s\n' "$$" > "$runner/runner_pid.txt"
trap 'code=$?;if ((code!=0));then cp "$runner/phase.txt" "$runner/failed_phase.txt";printf "failed\n" > "$runner/phase.txt";printf "%s\n" "$code" > "$runner/exit_code.txt";fi' EXIT
for mode in native cpu4 gpu4;do
 printf '%s\n' "$mode" > "$runner/phase.txt"
 timeout -k 30s 3600s env -u PYTHONPATH python3 -u "$base/run_service_trial.py" --output "/workspace/benchmark-results/20261020_cold_start_v2_service_${mode}_tianshu" --mode "$mode" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/${mode}.log" 2>&1
done
printf 'complete\n' > "$runner/phase.txt";date -Is > "$runner/complete.txt"
