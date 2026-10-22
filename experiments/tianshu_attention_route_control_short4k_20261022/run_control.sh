#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
sha256sum --check --status experiments/tianshu_attention_route_control_short4k_20261022/EXECUTION.sha256
python3 experiments/tianshu_attention_route_control_short4k_20261022/run_service_trial.py --output /workspace/benchmark-results/20261022_attention_route_control_short4k_tianshu --mode control --scope prefill --input-len 4096 --stop-at-utc 2026-10-22T15:58:00+00:00
