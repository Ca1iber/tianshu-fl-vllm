#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_native_dotsaddz_formats_20261023
sha256sum --check --status "$base/EXECUTION.sha256"
remaining=$(( $(date -u -d '2026-10-23 15:58:00 UTC' +%s) - $(date -u +%s) ));test "$remaining" -gt 2400
ixsmi | grep 'No running processes found'
runner=/workspace/benchmark-results/20261023_native_dotsaddz_formats_runner_tianshu
mkdir "$runner"
mapfile -t variants < <(jq -r '.variants[].name' "$base/CPU_GATE_32g.json")
for variant in "${variants[@]}";do
 ixsmi | grep 'No running processes found'
 if timeout --signal=TERM --kill-after=10s 45s python3 "$base/probe_gpu.py" --variant "$variant" > "$runner/${variant}.log" 2>&1;then status=0;else status=$?;fi
 printf '%s exit=%s\n' "$variant" "$status" >> "$runner/process_status.txt"
 ixsmi > "$runner/${variant}_hardware_after.txt"
 grep 'No running processes found' "$runner/${variant}_hardware_after.txt"
done
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 python3 "$base/verify_cpu.py" > "$runner/independent_CPU.log" 2>&1
sha256sum --check --status "$base/EXECUTION.sha256"
