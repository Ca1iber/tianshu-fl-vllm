#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_RMS_LM_formal_after_short_20261023
sha256sum --check --status "$base/EXECUTION.sha256"
while [ -e /proc/27948/stat ] && [ "$(awk '{print $3}' /proc/27948/stat)" != Z ];do
 test "$(date -u +%s)" -lt "$(date -u -d '2026-10-23 15:58:00 UTC' +%s)"
 sleep 10
done
short=experiments/tianshu_RMS_LM_short4k_20261023/SCREEN_GATE_prefill_32g.json
if ! test -f "$short" || ! jq -e '.complete and .candidate_passes_perf_screen and .vs_exact_r3_short_gain_percent>0.5 and .arms.candidate.hot_spread_pct<1 and .all_runtime_audits_independently_recomputed and .full_profile_audit_recomputed_exact and .checkpoint_SHA256_verified' "$short" >/dev/null;then
 printf '%s\n' 'COMBINED_SHORT_INCOMPLETE_OR_NOT_PASSED_NO_FORMAL_NO_PROMOTION'
 exit 0
fi
jq -e '.complete and .model_quality_qualified and .model_source_count==944 and .controller_count==91 and .new_combined_RMS_LM_model and .full_profile_and_accuracy_audit_recomputed_exact and .C8_correct/105>0.95 and .C64_correct/105>0.95' experiments/tianshu_RMS_LM_quality_results_20261023/QUALITY_GATE_32g.json >/dev/null
sha256sum --check --status experiments/tianshu_RMS_LM_quality_20261023/EXECUTION.sha256
sha256sum --check --status experiments/tianshu_RMS_LM_short4k_20261023/EXECUTION.sha256
sha256sum --check --status experiments/tianshu_RMS_LM_official4k_20261023/EXECUTION.sha256
sha256sum --check --status experiments/tianshu_RMS_LM_official_results_20261023/EXECUTION.sha256
remaining=$(( $(date -u -d '2026-10-23 15:58:00 UTC' +%s) - $(date -u +%s) ))
available=$(df -B1 --output=avail /workspace | tail -n 1)
if [ "$remaining" -le 2400 ] || [ "$available" -lt 1073741824 ];then
 printf '%s\n' 'COMBINED_SHORT_POSITIVE_FORMAL_NOT_STARTED_INSUFFICIENT_TIME_OR_DISK'
 exit 0
fi
if pgrep -f '^python3 experiments/.*/run_validation.py |^python3 experiments/.*/run_service_trial.py |^/usr/local/bin/python3.12 /usr/local/bin/vllm serve ' >/dev/null;then
 printf '%s\n' 'OTHER_GPU_TRIAL_ACTIVE_FORMAL_NOT_STARTED'
 exit 1
fi
ixsmi | grep 'No running processes found'
printf '%s\n' 'FRESH_COMBINED_QUALITY_AND_INDEPENDENT_SHORT_PASSED_STARTING_ORIGINAL_N256_FOUR_ROUND_4K'
bash experiments/tianshu_RMS_LM_official4k_20261023/run.sh > /workspace/benchmark-results/20261023_RMS_LM_official4k_runner_tianshu.log 2>&1
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 experiments/tianshu_RMS_LM_official_results_20261023/verify_cpu.py > /workspace/benchmark-results/20261023_RMS_LM_official4k_independent_CPU_tianshu.log 2>&1
gzip -n -9 -c experiments/tianshu_RMS_LM_official_results_20261023/CPU_COMPARISON_32g.json > experiments/tianshu_RMS_LM_official_results_20261023/CPU_COMPARISON_32g.json.gz
jq '{complete,promotion_passed,total_tok_s,output_tok_s,ttft_ms,hot_spread_percent,C8_correct,C64_correct,adopted}' experiments/tianshu_RMS_LM_official_results_20261023/OFFICIAL_GATE_32g.json
printf '%s\n' 'ORIGINAL_FORMAL_AND_INDEPENDENT_CPU_COMPLETE_NO_AUTOMATIC_RELEASE'
