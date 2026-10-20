#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_k8vi8_pv_int8_20261021
validation=/workspace/benchmark-results/20261021_dense_budget_v2_validation_runner_tianshu
runner=/workspace/benchmark-results/20261021_k8vi8_pv_int8_primitive_runner_tianshu
raw=/workspace/benchmark-results/20261021_k8vi8_pv_int8_primitive_GPU_tianshu
test ! -e "$runner";mkdir "$runner";printf 'waiting_for_validation\n' > "$runner/phase.txt"
trap 'code=$?;if ((code!=0));then cp "$runner/phase.txt" "$runner/failed_phase.txt";printf "failed\n" > "$runner/phase.txt";printf "%s\n" "$code" > "$runner/exit_code.txt";fi' EXIT
while :;do
 phase=$(cat "$validation/phase.txt")
 case "$phase" in complete|screen_regression|failed)break;; esac
 sleep 15
done
ixsmi > "$runner/hardware_before.txt"
if ! grep -q 'No running processes found' "$runner/hardware_before.txt";then printf 'GPU occupied after validation terminal state\n' > "$runner/reason.txt";exit 1;fi
printf 'gpu_primitive\n' > "$runner/phase.txt"
timeout -k 30s 900s env -u PYTHONPATH python3 -u "$base/probe_primitive_gpu.py" --output "$raw" --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/gpu.log" 2>&1
printf 'verifying_primitive\n' > "$runner/phase.txt"
env -u PYTHONPATH python3 "$base/verify_primitive.py" --root "$raw" > "$runner/verify.log" 2>&1
ixsmi > "$runner/hardware_after.txt"
grep -q 'No running processes found' "$runner/hardware_after.txt"
printf 'complete\n' > "$runner/phase.txt";date -Is > "$runner/complete.txt"
