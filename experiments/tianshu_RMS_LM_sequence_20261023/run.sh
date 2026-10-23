#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_RMS_LM_sequence_20261023
sha256sum --check --status "$base/EXECUTION.sha256"
while [ -e /proc/26165/stat ] && [ "$(awk '{print $3}' /proc/26165/stat)" != Z ];do
 test "$(date -u +%s)" -lt "$(date -u -d '2026-10-23 15:58:00 UTC' +%s)"
 sleep 10
done
for batch in 64 8;do
 result="/workspace/benchmark-results/20261023_RMS_LM_math${batch}_tianshu/result.json"
 if ! test -f "$result" || ! jq -e '.phase=="complete" and .accuracy_passed and .correct/105>0.95 and .evalscope_exit_code==0' "$result" >/dev/null;then
  printf 'RMS_LM_C%s_INCOMPLETE_OR_QUALITY_FAILED_NO_BENCHMARK\n' "$batch"
  exit 1
 fi
done
sha256sum --check --status experiments/tianshu_RMS_LM_quality_results_20261023/EXECUTION.sha256
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 experiments/tianshu_RMS_LM_quality_results_20261023/verify_cpu.py > /workspace/benchmark-results/20261023_RMS_LM_dual_quality_independent_CPU_tianshu.log 2>&1
jq -e '.complete and .model_quality_qualified and .model_source_count==944 and .controller_count==91 and .new_combined_RMS_LM_model and .all_inherited_and_new_runtime_audits_independently_recomputed and .no_fictitious_old_quantizer_calls and .old_LM64_execution_not_claimed' experiments/tianshu_RMS_LM_quality_results_20261023/QUALITY_GATE_32g.json >/dev/null
printf '%s\n' 'RMS_LM_FRESH_DUAL_QUALITY_PASSED_STARTING_SHORT_4K'
bash experiments/tianshu_RMS_LM_short4k_20261023/run_pair.sh > /workspace/benchmark-results/20261023_RMS_LM_short4k_pipeline_tianshu.log 2>&1
if jq -e '.complete and .candidate_passes_perf_screen' experiments/tianshu_RMS_LM_short4k_20261023/SCREEN_GATE_prefill_32g.json >/dev/null;then
 printf '%s\n' 'RMS_LM_SHORT_POSITIVE_PENDING_SEPARATE_FORMAL_FREEZE_AND_REVIEW'
else
 printf '%s\n' 'RMS_LM_SHORT_NOT_PASSED_NO_FORMAL_NO_PROMOTION'
fi
