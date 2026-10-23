#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_prefill_RMS_A8_QKV_20261023
sha256sum --check --status "$base/GPU_INPUTS.sha256"
for pid in 26165 27948 30742;do
 if [ -e "/proc/$pid/stat" ] && [ "$(awk '{print $3}' "/proc/$pid/stat")" != Z ];then printf 'Current model pipeline %s still live; new component not started\n' "$pid" >&2;exit 1;fi
done
if pgrep -f '^python3 experiments/.*/run_validation.py |^python3 experiments/.*/run_service_trial.py |^/usr/local/bin/python3.12 /usr/local/bin/vllm serve ' >/dev/null;then exit 1;fi
ixsmi | grep 'No running processes found'
available=$(df -B1 --output=avail /workspace | tail -n 1);test "$available" -ge 2469606196
remaining=$(( $(date -u -d '2026-10-23 15:58:00 UTC' +%s) - $(date -u +%s) ));test "$remaining" -gt 1200
root=/workspace/benchmark-results/20261023_prefill_RMS_A8_QKV_GPU_tianshu
runner=/workspace/benchmark-results/20261023_prefill_RMS_A8_QKV_runner_tianshu
mkdir "$runner"
finish(){ if [ -d "$root" ] && [ ! -f "$root/hardware_after.txt" ];then ixsmi > "$runner/hardware_after.txt";cp "$runner/hardware_after.txt" "$root/hardware_after.txt";fi; }
trap finish EXIT
timeout --signal=TERM --kill-after=10s "${remaining}s" env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 "$base/probe_gpu.py" --output "$root" --stop-at-utc 2026-10-23T15:58:00+00:00 > "$runner/GPU.log" 2>&1
for _ in $(seq 1 60);do
 ixsmi > "$runner/hardware_after.txt"
 if grep -q 'No running processes found' "$runner/hardware_after.txt";then break;fi
 sleep 1
done
grep -q 'No running processes found' "$runner/hardware_after.txt"
cp "$runner/hardware_after.txt" "$root/hardware_after.txt"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 "$base/verify_cpu.py" --root "$root" > "$runner/independent_CPU.log" 2>&1
sha256sum --check --status "$base/GPU_INPUTS.sha256"
