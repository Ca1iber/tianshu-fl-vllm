#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
for pid in 76900 80611;do
 while test -e "/proc/$pid";do
   state=$(awk '{print $3}' "/proc/$pid/stat");test "$state" = Z && break
   sleep 20
 done
done
ixsmi | grep -q 'No running processes found'
remaining=$(( $(date -u -d '2026-10-20 15:58:00 UTC' +%s)-$(date -u +%s) ))
test "$remaining" -gt 1200
limit=1800;if ((remaining<limit));then limit=$remaining;fi
base=experiments/tianshu_sql_target_reflection_20261020
root=/workspace/benchmark-results/20261020_sql_target_reflection_CPU_v2_tianshu
test ! -e "$root"
timeout -k 10s "${limit}s" env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -u "$base/probe_cpu.py"
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 "$base/verify_cpu.py"
