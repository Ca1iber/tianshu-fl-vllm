#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_IR_RMS_A8_chain_20261023
sha256sum --check --status "$base/EXECUTION.sha256"
ixsmi | grep 'No running processes found'
if pgrep -f '^python3 experiments/.*/run_validation.py |^python3 experiments/.*/run_service_trial.py |^/usr/local/bin/python3.12 /usr/local/bin/vllm serve ' >/dev/null;then exit 1;fi
remaining=$(( $(date -u -d '2026-10-23 15:58:00 UTC' +%s) - $(date -u +%s) ));test "$remaining" -gt 600
available=$(df -B1 --output=avail /workspace | tail -n 1);test "$available" -ge 1073741824
runner=/workspace/benchmark-results/20261023_IR_RMS_A8_chain_runner_tianshu
mkdir "$runner"
trap 'ixsmi > "$runner/hardware_after_exit.txt"' EXIT
ixsmi > "$runner/hardware_before.txt"
export VLLM_PLUGINS=fl OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
timeout --signal=TERM --kill-after=10s "${remaining}s" python3 -u "$base/probe_gpu.py" > "$runner/GPU.log" 2>&1
ixsmi > "$runner/hardware_after_GPU.txt"
grep -q 'No running processes found' "$runner/hardware_after_GPU.txt"
CUDA_VISIBLE_DEVICES='' python3 "$base/verify_cpu.py" > "$runner/independent_CPU.log" 2>&1
sha256sum --check --status "$base/EXECUTION.sha256"
printf '%s\n' 'IR_RMS_A8_CHAIN_COMPONENT_DONE_NO_MODEL_ADOPTION'
