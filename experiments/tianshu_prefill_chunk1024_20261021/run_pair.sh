#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_prefill_chunk1024_20261021
runner=/workspace/benchmark-results/20261021_prefill_chunk1024_pair_runner_tianshu
test -f /workspace/benchmark-results/20261021_sampler_exp_service_pair_runner_tianshu/complete.txt
test ! -e "$runner";mkdir "$runner";printf 'checking_idle\n' > "$runner/phase.txt"
trap 'code=$?;if ((code!=0));then cp "$runner/phase.txt" "$runner/failed_phase.txt";printf "failed\n" > "$runner/phase.txt";printf "%s\n" "$code" > "$runner/exit_code.txt";fi' EXIT
ixsmi > "$runner/hardware_before.txt";grep -q 'No running processes found' "$runner/hardware_before.txt"
for mode in control candidate;do
 printf '%s\n' "$mode" > "$runner/phase.txt"
 timeout -k 30s 3600s env -u PYTHONPATH python3 -u "$base/run_service_trial.py" --output "/workspace/benchmark-results/20261021_prefill_chunk1024_${mode}_tianshu" --mode "$mode" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/${mode}.log" 2>&1
done
printf 'verifying_pair\n' > "$runner/phase.txt"
env -u PYTHONPATH python3 "$base/verify_pair.py" > "$runner/verify.log" 2>&1
printf 'complete\n' > "$runner/phase.txt";date -Is > "$runner/complete.txt"
