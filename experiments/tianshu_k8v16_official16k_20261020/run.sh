#!/usr/bin/env bash
set -euo pipefail
repo=/workspace/vllm-plugin-FL
root=/workspace/benchmark-results/20261020_k8v16_official16k_tianshu
launcher=/workspace/benchmark-results/20261020_k8v16_official16k_launcher_tianshu
server_pid=
mkdir "$launcher"
test ! -e "$root"
finish() {
    trap - EXIT INT TERM
    if test -n "$server_pid" && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" || true
    fi
    ixsmi > "$launcher/hardware_after.txt"
    date --iso-8601=seconds > "$launcher/ended.txt"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ixsmi > "$launcher/hardware_before.txt"
grep -q 'No running processes found' "$launcher/hardware_before.txt"
sha256sum "$0" "$repo/benchmarks/benchmark_throughput_serve.py" "$repo/experiments/tianshu_k8v16_budget4096_release_20261019/serve_profile_template.json" > "$launcher/input_sources.sha256"
git -C "$repo" rev-parse HEAD > "$launcher/commit.txt"
git -C "$repo" status --short > "$launcher/git_status_before.txt"
date --iso-8601=seconds > "$launcher/started.txt"
env TIANSHU_K8V16_PROFILE_RECORD_DIR="$root" timeout -k 30s 7200s bash "$repo/experiments/tianshu_k8v16_budget4096_release_20261019/serve_tianshu_minicpm.sh" > "$launcher/server.log" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" > "$launcher/server_pid.txt"
ready=0
for attempt in $(seq 1 400); do
    kill -0 "$server_pid"
    if curl --fail --silent --show-error --max-time 3 http://127.0.0.1:9031/v1/models > "$launcher/models.json" 2>/dev/null; then
        ready=1
        break
    fi
    sleep 3
done
test "$ready" = 1
ln -s "$launcher/server.log" "$root/server.log"
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:9031/metrics > "$root/before_metrics.txt"
date --iso-8601=seconds > "$root/benchmark_started.txt"
cd /workspace
env VLLM_PLUGINS=fl PYTHONUNBUFFERED=1 timeout -k 20s 5400s python3 "$repo/benchmarks/benchmark_throughput_serve.py" \
    --model /workspace/MiniCPM5-2B \
    --served-model-name minicpm \
    --port 9031 \
    --test-cases '[[16384,1024,64,128]]' > "$root/benchmark.log" 2>&1
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:9031/metrics > "$root/after_metrics.txt"
date --iso-8601=seconds > "$root/benchmark_ended.txt"
sha256sum --check "$launcher/input_sources.sha256" > "$root/source_check_after.txt"
