#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
quant_runner=/workspace/benchmark-results/20261022_native_int8_prefill_runner_tianshu
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 experiments/tianshu_native_int8_prefill_20261022/probe_gpu.py --output /workspace/benchmark-results/20261022_native_int8_prefill_GPU_tianshu --stop-at-utc 2026-10-22T15:58:00+00:00 > "$quant_runner/gpu.log" 2>&1 &
printf '%s\n' "$!" > "$quant_runner/gpu_pid"
wait "$(cat "$quant_runner/gpu_pid")"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 experiments/tianshu_native_int8_prefill_20261022/verify_cpu.py --input /workspace/benchmark-results/20261022_native_int8_prefill_GPU_tianshu > "$quant_runner/cpu_audit.log" 2>&1
cp /workspace/benchmark-results/20261022_native_int8_prefill_GPU_tianshu/result.json experiments/tianshu_native_int8_prefill_20261022/GPU_GATE_32g.json
