#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_long_segments_validation_20261020
pipeline=/workspace/benchmark-results/20261020_long_segments_validation_pipeline_tianshu
screen=/workspace/benchmark-results/20261020_long_segments_service16k_tianshu
math=/workspace/benchmark-results/20261020_long_segments_math_tianshu
official=/workspace/benchmark-results/20261020_long_segments_official4k16k_tianshu
mkdir "$pipeline"
sha256sum "$base"/{run_validation.py,validation_profile_long.py,audit_profile.py,batch_policy.json,run_math_and_official.sh} > "$pipeline/frozen_entrypoints.sha256"
printf '%s\n' "$$" > "$pipeline/runner_pid.txt"
date --iso-8601=seconds > "$pipeline/started.txt"
env -u PYTHONPATH VLLM_PLUGINS=fl timeout -k 30s 7200s python3 -u "$base/run_validation.py" \
 --stage math --output "$math" --screen-root "$screen" --stop-at-utc 2026-10-20T15:58:00+00:00 \
 > "$pipeline/math.log" 2>&1
sha256sum --check "$pipeline/frozen_entrypoints.sha256"
# The math runner returns nonzero for <=95% accuracy, so set -e stops here.
env -u PYTHONPATH VLLM_PLUGINS=fl timeout -k 30s 7200s python3 -u "$base/run_validation.py" \
 --stage official --output "$official" --screen-root "$screen" --math-root "$math" \
 --stop-at-utc 2026-10-20T15:58:00+00:00 > "$pipeline/official.log" 2>&1
sha256sum --check "$pipeline/frozen_entrypoints.sha256"
date --iso-8601=seconds > "$pipeline/complete.txt"
