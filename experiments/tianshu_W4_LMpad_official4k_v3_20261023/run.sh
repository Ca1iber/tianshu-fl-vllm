#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_W4_LMpad_official4k_v3_20261023
sha256sum --check --status "$base/EXECUTION.sha256"
jq -e '.complete and .model_quality_qualified and .C8_correct>=100 and .C64_correct>=100' experiments/tianshu_W4_LMpad_quality_results_v3_20261023/QUALITY_GATE_32g.json >/dev/null
jq -e '.complete and .candidate_passes_perf_screen' experiments/tianshu_W4_LMpad_short4k_v3_20261023/SCREEN_GATE_prefill_32g.json >/dev/null
bash experiments/tianshu_native_int8_attention_prefill_release_20261022/restore_frozen_cache.sh --check
python3 "$base/run_validation.py" --output /workspace/benchmark-results/20261023_W4_LMpad_v3_official4k_tianshu --screen-root /workspace/vllm-plugin-FL/experiments/tianshu_native_int8_attention42_20261022 --stop-at-utc 2026-10-23T15:58:00+00:00 --stage official --batch 8 --down-scope prefill --math-root /workspace/benchmark-results/20261023_W4_LMpad_C8_CPU_recovery_tianshu --math64-root /workspace/benchmark-results/20261023_W4_LMpad_v2_math64_tianshu
