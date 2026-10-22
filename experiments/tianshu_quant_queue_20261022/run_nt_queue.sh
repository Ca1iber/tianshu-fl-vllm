#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
quant_queue=/workspace/benchmark-results/20261022_quant_nt_serial_queue_tianshu
mkdir -p "$quant_queue"
state() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$quant_queue/state.txt"; }
trap 'code=$?; state "EXIT $code"' EXIT
quant_previous_pid=$(cat /workspace/benchmark-results/20261022_quant_serial_queue_tianshu/pid)
state "WAIT_PREVIOUS $quant_previous_pid"
while test -r "/proc/$quant_previous_pid/stat"; do
 quant_process_state=$(awk '{print $3}' "/proc/$quant_previous_pid/stat")
 if test "$quant_process_state" = Z; then break; fi
 if test "$(date -u +%s)" -ge "$(date -ud '2026-10-22 15:58:00' +%s)"; then exit 2; fi
 sleep 5
done
grep -q QUEUE_COMPLETE /workspace/benchmark-results/20261022_quant_serial_queue_tianshu/state.txt
jq -e '.complete==true' experiments/tianshu_w8a16_postscale_20261022/INDEPENDENT_CPU_GATE_32g.json
jq -e '.complete==true' experiments/tianshu_w4a16_group_postscale_20261022/INDEPENDENT_CPU_GATE_32g.json
for quant_experiment in tianshu_w8a16_nt_postscale_20261022 tianshu_w4a16_nt_group_postscale_20261022; do
 case "$quant_experiment" in
  tianshu_w8a16_nt_postscale_20261022) quant_raw=20261022_w8a16_nt_postscale_GPU_tianshu; quant_runner=20261022_w8a16_nt_postscale_runner_tianshu ;;
  tianshu_w4a16_nt_group_postscale_20261022) quant_raw=20261022_w4a16_nt_group_postscale_GPU_tianshu; quant_runner=20261022_w4a16_nt_group_postscale_runner_tianshu ;;
 esac
 test "$(date -u +%s)" -lt "$(date -ud '2026-10-22 15:28:00' +%s)"
 ixsmi > "$quant_queue/hardware_before_$quant_experiment.txt"
 grep -q 'No running processes found' "$quant_queue/hardware_before_$quant_experiment.txt"
 mkdir -p "/workspace/benchmark-results/$quant_runner"
 state "GPU $quant_experiment"
 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 "experiments/$quant_experiment/probe_gpu.py" --output "/workspace/benchmark-results/$quant_raw" --stop-at-utc 2026-10-22T15:58:00+00:00 > "/workspace/benchmark-results/$quant_runner/gpu.log" 2>&1 &
 quant_child_pid=$!
 printf '%s\n' "$quant_child_pid" > "/workspace/benchmark-results/$quant_runner/pid"
 wait "$quant_child_pid"
 jq -e '.complete==true and .phase=="complete"' "/workspace/benchmark-results/$quant_raw/result.json"
 state "AUDIT $quant_experiment"
 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 "experiments/$quant_experiment/verify_cpu.py" --input "/workspace/benchmark-results/$quant_raw" > "/workspace/benchmark-results/$quant_runner/cpu_audit.log" 2>&1
 cp "/workspace/benchmark-results/$quant_raw/result.json" "experiments/$quant_experiment/GPU_GATE_32g.json"
 state "COMPLETE $quant_experiment"
done
state QUEUE_COMPLETE
