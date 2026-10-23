#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
sha256sum --check --status experiments/tianshu_QKV64_keep128_sequence_20261023/EXECUTION.sha256
while [ -e /proc/7900/stat ] && [ "$(awk '{print $3}' /proc/7900/stat)" != Z ];do
 test "$(date -u +%s)" -lt "$(date -u -d '2026-10-23 15:58:00 UTC' +%s)"
 sleep 10
done
for batch in 64 8;do
 result="/workspace/benchmark-results/20261023_QKV64_keep128_math${batch}_tianshu/result.json"
 if ! test -f "$result" || ! jq -e '.phase=="complete" and .accuracy_passed and .correct/105>0.95 and .evalscope_exit_code==0' "$result" >/dev/null;then
  printf 'KEEP128_C%s_INCOMPLETE_OR_QUALITY_FAILED_NO_BENCHMARK\n' "$batch"
  exit 1
 fi
done
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 experiments/tianshu_QKV64_keep128_quality_results_20261023/verify_cpu.py > /workspace/benchmark-results/20261023_QKV64_keep128_dual_quality_independent_CPU_tianshu.log 2>&1
jq -e '.complete and .model_quality_qualified and .model_source_count==928 and .controller_count==51 and .QKV128_native_retained_verified and .nonuniform_M64_BF16_policy_verified' experiments/tianshu_QKV64_keep128_quality_results_20261023/QUALITY_GATE_32g.json >/dev/null
printf '%s\n' 'KEEP128_DUAL_QUALITY_PASSED_STARTING_SHORT_4K'
bash experiments/tianshu_QKV64_keep128_short4k_20261023/run_pair.sh > /workspace/benchmark-results/20261023_QKV64_keep128_short4k_pipeline_tianshu.log 2>&1
if ! jq -e '.complete and .candidate_passes_perf_screen' experiments/tianshu_QKV64_keep128_short4k_20261023/SCREEN_GATE_prefill_32g.json >/dev/null;then
 printf '%s\n' 'KEEP128_SHORT_NEGATIVE_NO_FORMAL_NO_PROMOTION'
 exit 0
fi
printf '%s\n' 'KEEP128_SHORT_POSITIVE_STARTING_ORIGINAL_OFFICIAL_4K'
bash experiments/tianshu_QKV64_keep128_official4k_20261023/run.sh > /workspace/benchmark-results/20261023_QKV64_keep128_official4k_runner_tianshu.log 2>&1
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 experiments/tianshu_QKV64_keep128_official_results_20261023/verify_cpu.py > /workspace/benchmark-results/20261023_QKV64_keep128_official4k_independent_CPU_tianshu.log 2>&1
gzip -n -9 -c experiments/tianshu_QKV64_keep128_official_results_20261023/CPU_COMPARISON_32g.json > experiments/tianshu_QKV64_keep128_official_results_20261023/CPU_COMPARISON_32g.json.gz
printf '%s\n' 'KEEP128_FORMAL_AND_CPU_DONE_PENDING_REVIEW_NO_AUTO_ADOPTION'
date -Is
