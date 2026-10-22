#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
exec python3 experiments/tianshu_native_int8_prefill_service_20261022/run_smoke.py --mode candidate --output /workspace/benchmark-results/20261022_native_int8_prefill_service_smoke_candidate_tianshu --stop-at-utc 2026-10-22T15:58:00+00:00
