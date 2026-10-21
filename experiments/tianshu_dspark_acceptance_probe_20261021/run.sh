#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm-plugin-FL
base=/workspace/vllm-plugin-FL/experiments/tianshu_dspark_acceptance_probe_20261021
raw=/workspace/benchmark-results/20261021_dspark_acceptance_probe_GPU_tianshu
mkdir "$raw"
while IFS= read -r source_path; do
    expected=$(jq -r --arg path "$source_path" '.sources[$path]' "$base/CPU_GATE.json")
    actual=$(sha256sum "$source_path"); actual=${actual%% *}
    test "$actual" = "$expected"
    install -Dm644 "$source_path" "$raw/source_snapshot$source_path"
done < <(jq -r '.sources|keys[]' "$base/CPU_GATE.json")
install -Dm644 "$base/CPU_GATE.json" "$raw/source_snapshot$base/CPU_GATE.json"
cp "$(jq -r .inputs_path "$base/CPU_GATE.json")" "$raw/inputs.json"
mkdir "$raw/draft_config_alias"
jq 'del(.auto_map) | .dflash_config={target_layer_ids:.target_layer_ids,mask_token_id:.mask_token_id,block_size:7,projector_type:"dspark",use_aux_hidden_state:true,causal:false}' /workspace/MiniCPM5-2B-DSpark/config.json > "$raw/draft_config_alias/config.json"
ln -s /workspace/MiniCPM5-2B-DSpark/model.safetensors "$raw/draft_config_alias/model.safetensors"
export VLLM_PLUGINS=fl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export PYTHONPATH="$base/hooks"
export TRITON_PRINT_AUTOTUNING=1
export TIANSHU_DSPARK_COMPAT=1 TIANSHU_DSPARK_ZERO_VERIFY=1 TIANSHU_DSPARK_SAMPLE_VIEW=1
export TIANSHU_DSPARK_VERIFY_TILE=1 TIANSHU_DSPARK_MATCHED_WARMUP=1 TIANSHU_DSPARK_PREFILL_TILE=1
export TIANSHU_DSPARK_FIXED_INDEX=1 TIANSHU_DSPARK_FP32_THRESHOLDS=1
export TIANSHU_DSPARK_RECORD_DIR="$raw/adapter_records" TIANSHU_DSPARK_ACCEPTANCE_READY_FILE="$raw/READY_FOR_REQUESTS"
export VLLM_FL_USE_FLAGGEMS_ATTN=0 VLLM_FL_ILUVATAR_HYBRID_ATTENTION=0
export VLLM_FL_ILUVATAR_ADAPTIVE_DECODE_SEGMENTS=0 VLLM_FL_ILUVATAR_PREFILL_BUCKETS=0 VLLM_FL_ILUVATAR_LOGITS_GATHER=0
export FLAGGEMS_ILUVATAR_COMPACT_VARLEN=1
unset TIANSHU_NATIVE_PREFILL_PROBE TIANSHU_FA2_TILE_PROBE TIANSHU_PACK_PROBE_MODE
ixsmi > "$raw/hardware_before.txt"
grep -q 'No running processes found' "$raw/hardware_before.txt"
stop_epoch=$(date -u -d '2026-10-21 15:58:00' +%s)
remaining=$((stop_epoch-$(date -u +%s)))
test "$remaining" -gt 0
if test "$remaining" -gt 2400; then remaining=2400; fi
spec=$(jq -cn --arg model "$raw/draft_config_alias" '{method:"dflash",model:$model,num_speculative_tokens:2,draft_sample_method:"probabilistic",rejection_sample_method:"standard",enforce_eager:true}')
server_pid=
cleanup() {
    code=$?
    trap - EXIT INT TERM
    if test -n "$server_pid"; then
        kill -TERM -- "-$server_pid" 2>/dev/null || true
        for attempt in $(seq 1 20); do
            if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
            sleep 0.2
        done
        kill -KILL -- "-$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    jq -n --argjson exit_code "$code" --arg server_pid "$server_pid" '{complete:($exit_code==0),exit_code:$exit_code,server_pid:$server_pid,cleanup_attempted:true}' > "$raw/run_terminal.json"
    ixsmi > "$raw/hardware_after.txt"
    exit "$code"
}
trap cleanup EXIT INT TERM
jq -n --arg spec "$spec" --arg timeout "$remaining" '{phase:"starting",speculative_config:($spec|fromjson),timeout_seconds:($timeout|tonumber),primary_source_inventory:true,official_benchmark:false}' > "$raw/run_start.json"
setsid timeout --signal=TERM --kill-after=15s "${remaining}s" vllm serve /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm --gpu-memory-utilization .85 --max-model-len 131072 --max-num-batched-tokens 2048 --max-num-seqs 8 --speculative-config "$spec" --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' > "$raw/server.log" 2>&1 < /dev/null &
server_pid=$!
printf '%s\n' "$server_pid" > "$raw/server.pid"
ready=0
for attempt in $(seq 1 600); do
    if ! kill -0 "$server_pid" 2>/dev/null; then tail -30 "$raw/server.log"; exit 1; fi
    if curl -fsS --max-time 2 http://127.0.0.1:9031/v1/models > "$raw/models.json" 2>/dev/null && jq -e '.data|any(.id=="minicpm")' "$raw/models.json" > /dev/null; then ready=1; break; fi
    sleep 2
done
test "$ready" -eq 1
PYTHONPATH= timeout --signal=TERM --kill-after=10s 900s python3 -u "$base/client.py" --root "$raw" > "$raw/client.log" 2>&1
