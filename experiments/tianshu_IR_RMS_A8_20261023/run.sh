#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_IR_RMS_A8_20261023
sha256sum --check --status "$base/EXECUTION.sha256"
for pid in 7900 9812;do
 if [ -e "/proc/$pid/stat" ] && [ "$(awk '{print $3}' "/proc/$pid/stat")" != Z ];then printf 'Prior GPU pipeline %s still live; no component started\n' "$pid" >&2;exit 1;fi
done
if pgrep -f '^python3 experiments/.*/run_validation.py |^python3 experiments/.*/run_service_trial.py |^/usr/local/bin/python3.12 /usr/local/bin/vllm serve ' >/dev/null;then exit 1;fi
ixsmi | grep 'No running processes found'
remaining=$(( $(date -u -d '2026-10-23 15:58:00 UTC' +%s) - $(date -u +%s) ))
test "$remaining" -gt 600
available=$(df -B1 --output=avail /workspace | tail -n 1)
test "$available" -ge 1073741824
runner=/workspace/benchmark-results/20261023_IR_RMS_A8_runner_tianshu
mkdir "$runner"
trap 'ixsmi > "$runner/hardware_after_exit.txt"' EXIT
ixsmi > "$runner/hardware_before.txt"
export VLLM_PLUGINS=fl
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
timeout --signal=TERM --kill-after=10s "${remaining}s" python3 -u "$base/probe_gpu.py" --output /workspace/benchmark-results/20261023_IR_RMS_A8_GPU_tianshu --stop-at-utc 2026-10-23T15:58:00+00:00 > "$runner/GPU.log" 2>&1
for attempt in {1..30};do
 ixsmi > "$runner/hardware_after_GPU.txt"
 if grep -q 'No running processes found' "$runner/hardware_after_GPU.txt";then break;fi
 sleep 1
done
grep -q 'No running processes found' "$runner/hardware_after_GPU.txt"
CUDA_VISIBLE_DEVICES='' python3 "$base/verify_cpu.py" > "$runner/independent_CPU.log" 2>&1
sha256sum --check --status "$base/EXECUTION.sha256"
printf '%s\n' 'IR_RMS_A8_COMPONENT_GPU_AND_CPU_DONE_NO_MODEL_ADOPTION'
