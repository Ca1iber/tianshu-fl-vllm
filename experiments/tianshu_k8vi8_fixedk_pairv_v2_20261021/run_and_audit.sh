#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_k8vi8_fixedk_pairv_v2_20261021
root=/workspace/benchmark-results/20261021_k8vi8_fixedk_pairv_v2_GPU_tianshu
runner=/workspace/benchmark-results/20261021_k8vi8_fixedk_pairv_v2_runner_tianshu
bash "$base/run_component.sh"
printf 'CPU_rechecking\n' > "$runner/phase.txt"
timeout -k 20s 900s env -u PYTHONPATH CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 -u "$base/verify_gpu.py" --root "$root" > "$runner/CPU_verify.log" 2>&1
jq '[.cases[]|{batch,context,strided,segments,raw_ms,median_ms}]' "$root/result.json" > "$base/ROUNDS_32g.json"
printf 'complete\n' > "$runner/phase.txt";date -Is > "$runner/complete.txt"
