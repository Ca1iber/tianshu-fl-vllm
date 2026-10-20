#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
frozen=/workspace/vllm-plugin-FL/experiments/tianshu_prefill_budget8192_service_20261021
recovery=/workspace/vllm-plugin-FL/experiments/tianshu_prefill_budget8192_audit_results_20261021
runner=/workspace/benchmark-results/20261021_prefill_budget8192_recovery_runner_tianshu
mkdir -p "$runner"
trap 'printf "failed\n" > "$runner/phase.txt"' ERR
sha256sum --check "$recovery/CONTROLLERS_32g.sha256" > "$runner/controllers_before.txt"
printf 'candidate\n' > "$runner/phase.txt"
set +e
python3 "$frozen/run_service_trial.py" --mode candidate --output /workspace/benchmark-results/20261021_prefill_budget8192_service_candidate_tianshu --stop-at-utc '2026-10-21T15:58:00+00:00' > "$runner/candidate.log" 2>&1
status=$?
set -e
[ "$status" -eq 1 ]
jq -e '.phase=="audit_failed" and .error=="KeyError('\''native_sha256'\'')" and (.screen_rounds|length)==3' /workspace/benchmark-results/20261021_prefill_budget8192_service_candidate_tianshu/result.json > "$runner/expected_metadata_failure.txt"
printf 'CPU_recheck\n' > "$runner/phase.txt"
CUDA_VISIBLE_DEVICES=-1 python3 "$recovery/verify_pair.py" > "$runner/verify.log" 2>&1
sha256sum --check "$recovery/CONTROLLERS_32g.sha256" > "$runner/controllers_after.txt"
printf 'complete\n' > "$runner/phase.txt"
