#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_prefill_RMS_QKV_quality_v2_20261023
sha256sum --check --status "$base/EXECUTION.sha256"
if pgrep -f '^python3 experiments/.*/run_validation.py |^python3 experiments/.*/run_service_trial.py |^/usr/local/bin/python3.12 /usr/local/bin/vllm serve ' >/dev/null;then exit 1;fi
ixsmi | grep 'No running processes found'
remaining=$(( $(date -u -d '2026-10-23 15:58:00 UTC' +%s) - $(date -u +%s) ));test "$remaining" -gt 7200
available=$(df -B1 --output=avail /workspace | tail -n 1);test "$available" -ge 2147483648
jq -e '.complete and .component_lead' experiments/tianshu_prefill_RMS_A8_QKV_20261023/INDEPENDENT_CPU_GATE_32g.json >/dev/null
bash experiments/tianshu_native_int8_attention_prefill_release_20261022/restore_frozen_cache.sh --check
runner=/workspace/benchmark-results/20261023_prefill_RMS_QKV_v2_quality_runner_tianshu
mkdir "$runner"
for batch in 64 8;do
 python3 "$base/run_validation.py" --output "/workspace/benchmark-results/20261023_prefill_RMS_QKV_v2_math${batch}_tianshu" --screen-root /workspace/vllm-plugin-FL/experiments/tianshu_native_int8_attention42_20261022 --stop-at-utc 2026-10-23T15:58:00+00:00 --stage math --batch "$batch" --down-scope prefill > "$runner/math${batch}.log" 2>&1
done
python3 experiments/tianshu_prefill_RMS_QKV_quality_results_v2_20261023/verify_cpu.py > "$runner/independent_CPU.log" 2>&1
