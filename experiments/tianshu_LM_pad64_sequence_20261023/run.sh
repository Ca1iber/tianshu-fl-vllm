#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
sha256sum --check --status experiments/tianshu_LM_pad64_sequence_20261023/EXECUTION.sha256
while [ -e /proc/1807/stat ] && [ "$(awk '{print $3}' /proc/1807/stat)" != Z ];do
 test "$(date -u +%s)" -lt "$(date -u -d '2026-10-23 15:58:00 UTC' +%s)"
 sleep 10
done
for arm in control candidate;do jq -e '.phase=="complete" and (.screen_rounds|length)==3' "/workspace/benchmark-results/20261023_LM_pad64_short4k_${arm}_tianshu/result.json" >/dev/null;done
if ! jq -e '.complete and .candidate_passes_perf_screen' experiments/tianshu_LM_pad64_short4k_20261023/SCREEN_GATE_prefill_32g.json >/dev/null;then
 printf '%s\n' 'LM_SHORT_NEGATIVE_NO_FORMAL_NO_PROMOTION'
 exit 0
fi
printf '%s\n' 'LM_SHORT_POSITIVE_STARTING_ORIGINAL_OFFICIAL_4K'
bash experiments/tianshu_LM_pad64_official4k_20261023/run.sh > /workspace/benchmark-results/20261023_LM_pad64_official4k_runner_tianshu.log 2>&1
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 experiments/tianshu_LM_pad64_official_results_20261023/verify_cpu.py > /workspace/benchmark-results/20261023_LM_pad64_official4k_independent_CPU_tianshu.log 2>&1
gzip -n -9 -c experiments/tianshu_LM_pad64_official_results_20261023/CPU_COMPARISON_32g.json > experiments/tianshu_LM_pad64_official_results_20261023/CPU_COMPARISON_32g.json.gz
printf '%s\n' 'LM_FORMAL_AND_CPU_DONE_PENDING_REVIEW_NO_AUTO_ADOPTION'
date -Is
