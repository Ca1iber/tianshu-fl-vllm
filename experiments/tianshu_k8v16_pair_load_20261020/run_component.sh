#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
root=/workspace/benchmark-results/20261020_k8v16_pair_load_GPU_tianshu
runner=/workspace/benchmark-results/20261020_k8v16_pair_load_runner_tianshu
cutoff=$(date -u -d '2026-10-20 15:58:00 UTC' +%s)
remaining=$((cutoff-$(date -u +%s)))
test "$remaining" -gt 1200
test ! -e "$root"
test ! -e "$runner"
ixsmi | grep -q 'No running processes found'
mkdir "$runner"
printf '%s\n' "$$" > "$runner/runner_pid.txt"
printf '%s\n' component_running > "$runner/phase.txt"
trap 'component_exit=$?; if (( component_exit!=0 )); then cp "$runner/phase.txt" "$runner/failed_phase.txt"; printf "failed\n" > "$runner/phase.txt"; printf "%s\n" "$component_exit" > "$runner/exit_code.txt"; fi' EXIT
limit=1800
if (( remaining<limit )); then limit=$remaining; fi
timeout -k 30s "${limit}s" env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -u experiments/tianshu_k8v16_pair_load_20261020/probe_gpu.py --output "$root" --reader-source experiments/tianshu_k8v16_address32_20261020/qualified_S16_reader.py --stop-at-utc 2026-10-20T15:58:00+00:00 > "$runner/component.log" 2>&1
printf '%s\n' independent_CPU_verification > "$runner/phase.txt"
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 experiments/tianshu_k8v16_pair_load_20261020/verify_gpu.py "$root" > "$runner/verification.log" 2>&1
ixsmi > "$runner/hardware_after.txt"
grep -q 'No running processes found' "$runner/hardware_after.txt"
printf '%s\n' complete > "$runner/phase.txt"
date -Is > "$runner/complete.txt"
