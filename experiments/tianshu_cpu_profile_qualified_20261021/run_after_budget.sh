#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
previous=/workspace/benchmark-results/20261021_dense_budget4352_v1_pair_runner_tianshu
runner=/workspace/benchmark-results/20261021_cpu_profile_qualified_runner_tianshu
test ! -e "$runner";mkdir "$runner"
printf 'waiting_for_budget_terminal\n' > "$runner/phase.txt"
while :;do
 phase=$(cat "$previous/phase.txt")
 case "$phase" in complete|failed)break;;esac
 sleep 10
done
printf 'checking_gpu_idle\n' > "$runner/phase.txt"
ixsmi > "$runner/hardware_before.txt"
grep -q 'No running processes found' "$runner/hardware_before.txt"
printf 'profiling\n' > "$runner/phase.txt"
trap 'code=$?;if ((code!=0));then printf "failed\n" > "$runner/phase.txt";printf "%s\n" "$code" > "$runner/exit_code.txt";fi' EXIT
timeout -k 30s 1800s env -u PYTHONPATH python3 -u experiments/tianshu_cpu_profile_qualified_20261021/run_profile.py --output /workspace/benchmark-results/20261021_cpu_profile_qualified_tianshu --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/profile.log" 2>&1
printf 'complete\n' > "$runner/phase.txt";date -Is > "$runner/complete.txt"
