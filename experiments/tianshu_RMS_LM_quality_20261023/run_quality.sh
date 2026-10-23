#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_RMS_LM_quality_20261023
sha256sum --check --status "$base/EXECUTION.sha256"
for pid in 18102 19611;do
 if [ -e "/proc/$pid/stat" ] && [ "$(awk '{print $3}' "/proc/$pid/stat")" != Z ];then printf 'Previous RMS pipeline %s still active; combined GPU not started\n' "$pid" >&2;exit 1;fi
done
if pgrep -f '^python3 experiments/.*/run_validation.py |^python3 experiments/.*/run_service_trial.py |^/usr/local/bin/python3.12 /usr/local/bin/vllm serve ' >/dev/null;then exit 1;fi
ixsmi | grep 'No running processes found'
remaining=$(( $(date -u -d '2026-10-23 15:58:00 UTC' +%s) - $(date -u +%s) ));test "$remaining" -gt 7200
available=$(df -B1 --output=avail /workspace | tail -n 1);test "$available" -ge 2147483648
jq -e '.complete and .model_quality_qualified and .C8_correct/105>0.95 and .C64_correct/105>0.95' experiments/tianshu_RMS_A8_GU_quality_results_v2_20261023/QUALITY_GATE_32g.json >/dev/null
bash experiments/tianshu_native_int8_attention_prefill_release_20261022/restore_frozen_cache.sh --check
sha256sum --check --status "$base/LM_COMPONENT_CACHE.sha256"
runner=/workspace/benchmark-results/20261023_RMS_LM_quality_runner_tianshu
mkdir "$runner"
for batch in 64 8;do
 python3 "$base/run_validation.py" --output "/workspace/benchmark-results/20261023_RMS_LM_math${batch}_tianshu" --screen-root /workspace/vllm-plugin-FL/experiments/tianshu_LM_pad64_20261022 --stop-at-utc 2026-10-23T15:58:00+00:00 --stage math --batch "$batch" --down-scope prefill > "$runner/math${batch}.log" 2>&1
done
