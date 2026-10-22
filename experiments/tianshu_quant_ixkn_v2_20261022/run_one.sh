#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
quant_raw=/workspace/benchmark-results/20261022_ixkn_v2_down_M64_W8_channel_candidate_tianshu
mkdir "$quant_raw"
ixsmi > "$quant_raw/hardware_before.txt"
grep -q 'No running processes found' "$quant_raw/hardware_before.txt"
set +e
timeout --signal=INT --kill-after=30s 900s /usr/local/corex-4.5.0/bin/ixkn-cli --profile-from-start on --disable-profiler-start-stop --kernel-name _matmul --launch-skip 10 --launch-count 1 --set detailed --cache-control none --csv --log-file "$quant_raw/metrics.csv" python3 experiments/tianshu_quant_ixkn_20261022/profile_one.py --role down --m 64 --format W8_channel --path candidate --output "$quant_raw/app" --stop-at-utc 2026-10-22T15:58:00+00:00 > "$quant_raw/app.log" 2>&1 < /dev/null
quant_exit=$?
printf '%s\n' "$quant_exit" > "$quant_raw/profiler_exit.txt"
ixsmi > "$quant_raw/hardware_after.txt"
exit "$quant_exit"
