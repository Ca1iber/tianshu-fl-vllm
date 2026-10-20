#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=experiments/tianshu_sampler_exp_cold_v2_20261021
runner=/workspace/benchmark-results/20261021_sampler_exp_cold_v2_runner_tianshu
test -f /workspace/benchmark-results/20261021_prefill_chunk1024_pair_runner_tianshu/complete.txt
test ! -e "$runner";mkdir "$runner";printf 'component\n' > "$runner/phase.txt"
trap 'code=$?;if ((code!=0));then cp "$runner/phase.txt" "$runner/failed_phase.txt";printf "failed\n" > "$runner/phase.txt";printf "%s\n" "$code" > "$runner/exit_code.txt";fi' EXIT
timeout -k 20s 1200s env -u PYTHONPATH TRITON_PRINT_AUTOTUNING=1 python3 -u "$base/probe_gpu.py" --output /workspace/benchmark-results/20261021_sampler_exp_cold_v2_GPU_tianshu --stop-at-utc 2026-10-21T15:58:00+00:00 > "$runner/probe.log" 2>&1
printf 'CPU_recheck\n' > "$runner/phase.txt"
env -u PYTHONPATH python3 "$base/verify_gpu.py" --root /workspace/benchmark-results/20261021_sampler_exp_cold_v2_GPU_tianshu --log "$runner/probe.log" > "$runner/verify.log" 2>&1
printf 'complete\n' > "$runner/phase.txt";date -Is > "$runner/complete.txt"
