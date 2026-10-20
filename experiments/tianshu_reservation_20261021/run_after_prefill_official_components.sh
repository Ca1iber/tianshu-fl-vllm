#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
runner=/workspace/benchmark-results/20261021_after_prefill_components_runner_tianshu
parent=/workspace/benchmark-results/20261021_prefill_int8_validation_runner_tianshu
formal=/workspace/benchmark-results/20261021_prefill_int8_official4k_tianshu
math=/workspace/benchmark-results/20261021_prefill_int8_math_tianshu
deadline=$(date -u -d '2026-10-21 15:58:00 UTC' +%s)
test ! -e "$runner";mkdir "$runner";printf 'waiting_for_existing_official4k\n' > "$runner/phase.txt"
trap 'code=$?;if ((code!=0));then cp "$runner/phase.txt" "$runner/failed_phase.txt";printf "failed\n" > "$runner/phase.txt";printf "%s\n" "$code" > "$runner/exit_code.txt";fi' EXIT
while ! test -e "$parent/complete.txt"; do
    (( $(date -u +%s) < deadline-3600 ))
    phase=$(cat "$parent/phase.txt")
    case "$phase" in failed*|stopped*) printf '%s\n' "$phase" >&2;exit 1;;esac
    sleep 10
done
jq -e '.phase=="complete" and .completed_runs==4 and .benchmark_exit_code==0' "$formal/result.json" >/dev/null
jq -e '.complete and .total==105 and .accuracy>0.95' "$math/accuracy_audit.json" >/dev/null
printf 'down_component\n' > "$runner/phase.txt"
bash experiments/tianshu_int8_down_grouped_20261021/run_component.sh
printf 'down_CPU_recheck\n' > "$runner/phase.txt"
timeout -k 20s 900s env -u PYTHONPATH CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 -u experiments/tianshu_int8_down_grouped_results_20261021/verify_gpu.py /workspace/benchmark-results/20261021_int8_down_grouped_GPU_tianshu > "$runner/down_verify.log" 2>&1
printf 'complete\n' > /workspace/benchmark-results/20261021_int8_down_grouped_runner_tianshu/phase.txt
date -Is > /workspace/benchmark-results/20261021_int8_down_grouped_runner_tianshu/complete.txt
printf 'KV_fixedk_pairv_component\n' > "$runner/phase.txt"
bash experiments/tianshu_k8vi8_fixedk_pairv_20261021/run_component.sh
printf 'KV_CPU_recheck\n' > "$runner/phase.txt"
timeout -k 20s 900s env -u PYTHONPATH CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 -u experiments/tianshu_k8vi8_fixedk_pairv_20261021/verify_gpu.py --root /workspace/benchmark-results/20261021_k8vi8_fixedk_pairv_GPU_tianshu > "$runner/KV_verify.log" 2>&1
jq '[.cases[]|{batch,context,strided,segments,raw_ms,median_ms}]' /workspace/benchmark-results/20261021_k8vi8_fixedk_pairv_GPU_tianshu/result.json > experiments/tianshu_k8vi8_fixedk_pairv_20261021/ROUNDS_32g.json
printf 'complete\n' > /workspace/benchmark-results/20261021_k8vi8_fixedk_pairv_runner_tianshu/phase.txt
date -Is > /workspace/benchmark-results/20261021_k8vi8_fixedk_pairv_runner_tianshu/complete.txt
printf 'complete\n' > "$runner/phase.txt";date -Is > "$runner/complete.txt"
