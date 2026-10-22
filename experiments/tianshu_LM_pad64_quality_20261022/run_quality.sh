#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
sha256sum --check --status experiments/tianshu_LM_pad64_quality_20261022/EXECUTION.sha256
sha256sum --check --status experiments/tianshu_LM_pad64_quality_20261022/LM_COMPONENT_CACHE.sha256
available_bytes=$(df -B1 --output=avail /workspace | tail -n 1)
test "$available_bytes" -ge 2147483648
bash experiments/tianshu_native_int8_attention_prefill_release_20261022/restore_frozen_cache.sh --check
lm_runner=/workspace/benchmark-results/20261022_LM_pad64_quality_runner_tianshu
mkdir "$lm_runner"
for batch in 64 8; do
python3 experiments/tianshu_LM_pad64_quality_20261022/run_validation.py --output "/workspace/benchmark-results/20261022_LM_pad64_math${batch}_tianshu" --screen-root /workspace/vllm-plugin-FL/experiments/tianshu_LM_pad64_20261022 --stop-at-utc 2026-10-22T15:58:00+00:00 --stage math --batch "$batch" --down-scope prefill > "$lm_runner/math${batch}.log" 2>&1
done
