#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
sha256sum --check --status experiments/tianshu_native_int8_attention_decode_scope_sequence_20261022/EXECUTION.sha256
scope_deadline=$(date -u -d '2026-10-22 15:58:00 UTC' +%s)
scope_math8=/workspace/benchmark-results/20261022_native_int8_attention_decode_scope_math8_tianshu
scope_math64=/workspace/benchmark-results/20261022_native_int8_attention_decode_scope_math64_tianshu
while [ ! -f "$scope_math8/result.json" ]; do
 test "$(date -u +%s)" -lt "$scope_deadline"
 kill -0 106089
 sleep 15
done
jq -e '.phase=="complete" and .accuracy_passed==true and .correct/105>0.95' "$scope_math8/result.json" >/dev/null
jq -e '.phase=="complete" and .accuracy_passed==true and .correct/105>0.95' "$scope_math64/result.json" >/dev/null
date -Is
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 experiments/tianshu_native_int8_attention_decode_scope_quality_results_20261022/verify_cpu.py > /workspace/benchmark-results/20261022_attention_decode_scope_dual_quality_independent_CPU_tianshu.log 2>&1
jq -e '.complete and .model_quality_qualified and .model_source_count==928 and .controller_count==51 and .C8_correct/105>0.95 and .C64_correct/105>0.95 and .actual_new_QKV_M64_uniform_FULL_capture42_O_BF16_capture42 and .new_small_prefill_BF16_policy_verified' experiments/tianshu_native_int8_attention_decode_scope_quality_results_20261022/QUALITY_GATE_32g.json >/dev/null
printf '%s\n' 'DUAL_QUALITY_PASSED_STARTING_SHORT_4K'
bash experiments/tianshu_native_int8_attention_decode_scope_short4k_20261022/run_pair.sh > /workspace/benchmark-results/20261022_native_int8_attention_decode_scope_short4k_pipeline_tianshu.log 2>&1
if ! jq -e '.complete and .candidate_passes_perf_screen' experiments/tianshu_native_int8_attention_decode_scope_short4k_20261022/SCREEN_GATE_prefill_32g.json >/dev/null;then
 printf '%s\n' 'SHORT_SCREEN_NEGATIVE_NO_FORMAL_NO_PROMOTION'
 exit 0
fi
printf '%s\n' 'SHORT_SCREEN_PASSED_STARTING_ORIGINAL_OFFICIAL_4K'
bash experiments/tianshu_native_int8_attention_decode_scope_official4k_20261022/run.sh > /workspace/benchmark-results/20261022_native_int8_attention_decode_scope_official4k_runner_tianshu.log 2>&1
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 experiments/tianshu_native_int8_attention_decode_scope_official_results_20261022/verify_cpu.py > /workspace/benchmark-results/20261022_attention_decode_scope_official4k_independent_CPU_tianshu.log 2>&1
gzip -n -9 -c experiments/tianshu_native_int8_attention_decode_scope_official_results_20261022/CPU_COMPARISON_32g.json > experiments/tianshu_native_int8_attention_decode_scope_official_results_20261022/CPU_COMPARISON_32g.json.gz
printf '%s\n' 'FORMAL_AND_INDEPENDENT_CPU_COMPLETE_PENDING_REVIEW_NO_AUTO_ADOPTION'
date -Is
