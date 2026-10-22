#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
sha256sum --check --status experiments/tianshu_LM_pad64_sequence_20261022/EXECUTION.sha256
while [ ! -f /workspace/benchmark-results/20261022_LM_pad64_math8_tianshu/result.json ]; do
 test "$(date -u +%s)" -lt "$(date -u -d '2026-10-22 15:58:00 UTC' +%s)"
 sleep 10
done
while [ -e /proc/118164/stat ] && [ "$(awk '{print $3}' /proc/118164/stat)" != Z ]; do sleep 1; done
jq -e '.phase=="complete" and .accuracy_passed and .correct/105>0.95' /workspace/benchmark-results/20261022_LM_pad64_math8_tianshu/result.json >/dev/null
jq -e '.complete and .accuracy_passed and (.original_experiment_passed==false) and .new_actual_Math_guard_execution_traced and .correct/105>0.95' experiments/tianshu_LM_pad64_audit_recovery_20261022/RECOVERY_GATE_32g.json >/dev/null
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 experiments/tianshu_LM_pad64_quality_results_v2_20261022/verify_cpu.py > /workspace/benchmark-results/20261022_LM_pad64_dual_quality_independent_CPU_v2_tianshu.log 2>&1
jq -e '.complete and .model_quality_qualified and .model_source_count==925 and .controller_count==82 and (.C64_original_experiment_passed==false) and .C64_CPU_runtime_audit_recovered' experiments/tianshu_LM_pad64_quality_results_v2_20261022/QUALITY_GATE_32g.json >/dev/null
printf '%s\n' 'LM_DUAL_QUALITY_INDEPENDENT_PASSED_STARTING_SHORT_4K'
bash experiments/tianshu_LM_pad64_short4k_20261022/run_pair.sh > /workspace/benchmark-results/20261022_LM_pad64_short4k_pipeline_tianshu.log 2>&1
if ! jq -e '.complete and .candidate_passes_perf_screen' experiments/tianshu_LM_pad64_short4k_20261022/SCREEN_GATE_prefill_32g.json >/dev/null;then
 printf '%s\n' 'LM_SHORT_NEGATIVE_NO_FORMAL_NO_PROMOTION'
 exit 0
fi
printf '%s\n' 'LM_SHORT_POSITIVE_STARTING_ORIGINAL_OFFICIAL_4K'
bash experiments/tianshu_LM_pad64_official4k_20261022/run.sh > /workspace/benchmark-results/20261022_LM_pad64_official4k_runner_tianshu.log 2>&1
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 experiments/tianshu_LM_pad64_official_results_20261022/verify_cpu.py > /workspace/benchmark-results/20261022_LM_pad64_official4k_independent_CPU_tianshu.log 2>&1
gzip -n -9 -c experiments/tianshu_LM_pad64_official_results_20261022/CPU_COMPARISON_32g.json > experiments/tianshu_LM_pad64_official_results_20261022/CPU_COMPARISON_32g.json.gz
printf '%s\n' 'LM_FORMAL_AND_CPU_DONE_PENDING_REVIEW_NO_AUTO_ADOPTION'
date -Is
