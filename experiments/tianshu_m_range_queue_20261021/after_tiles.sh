#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
runner=/workspace/benchmark-results/20261021_m_range_serial_queue_tianshu
mkdir "$runner"
printf '%s\n' "$$" > "$runner/queue_pid.txt"
trap 'code=$?; printf "%s\n" "$code" > "$runner/exit_code.txt"; if (( code!=0 )); then cp "$runner/phase.txt" "$runner/failed_phase.txt"; printf "failed\n" > "$runner/phase.txt"; fi' EXIT
printf 'waiting_tiles_mlp42\n' > "$runner/phase.txt"
while true; do
 status=$(ps -p 275710 -o stat= || true)
 case "$status" in ''|Z*) break;; esac
 test "$(date -u +%s)" -lt "$(date -u -d '2026-10-21 15:30:00 UTC' +%s)"
 sleep 5
done
jq -e '.complete and .phase=="complete"' /workspace/benchmark-results/20261021_m_range_MLP42_tiles_v2_GPU_tianshu/result.json >/dev/null
printf 'verifying_tiles_mlp42_CPU\n' > "$runner/phase.txt"
env CUDA_VISIBLE_DEVICES='' VLLM_PLUGINS='' python3 experiments/tianshu_m_range_mlp42_tiles_v2_results_20261021/verify_cpu.py > "$runner/tiles_verification.log" 2>&1
sed -i 's/\r$//' experiments/tianshu_m_range_mlp42_tiles_v2_results_20261021/MLP42_TIMINGS_32g.csv
jq -e '.complete and .performance_valid' experiments/tianshu_m_range_mlp42_tiles_v2_results_20261021/GPU_GATE_32g.json >/dev/null
ixsmi | grep -q 'No running processes found'
test "$(date -u +%s)" -lt "$(date -u -d '2026-10-21 15:30:00 UTC' +%s)"
printf 'interleaved_single_dot\n' > "$runner/phase.txt"
root=/workspace/benchmark-results/20261021_nn_silu_interleaved_GPU_tianshu
test ! -e "$root"
python3 -u experiments/tianshu_nn_silu_interleaved_20261021/probe_gpu.py --output "$root" > "$runner/interleaved.log" 2>&1 &
child=$!
printf '%s\n' "$child" > "$runner/child_pid.txt"
wait "$child"
printf 'verifying_interleaved_CPU\n' > "$runner/phase.txt"
env CUDA_VISIBLE_DEVICES='' VLLM_PLUGINS='' python3 experiments/tianshu_nn_silu_interleaved_results_20261021/verify_cpu.py > "$runner/interleaved_verification.log" 2>&1
for file in experiments/tianshu_nn_silu_interleaved_results_20261021/*_32g.csv; do test ! -e "$file" || sed -i 's/\r$//' "$file"; done
printf 'complete\n' > "$runner/phase.txt"
