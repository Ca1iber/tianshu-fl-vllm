#!/usr/bin/env bash
set -euo pipefail

# Native Prefill +Decode KV16/w4/stage1 +75-point buckets +fixed gather.
# Independent Math102/105; original4K/16K results recorded separately.
# Quality checkpoint 26c0ba1.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VLLM_PLUGINS=fl
export VLLM_FL_ILUVATAR_HYBRID_ATTENTION=1
export FLAGGEMS_ILUVATAR_COMPACT_VARLEN=1
export VLLM_FL_ILUVATAR_ADAPTIVE_DECODE_SEGMENTS=1
export VLLM_FL_USE_FLAGGEMS_ATTN=0
export VLLM_FL_ILUVATAR_SPLIT_MIXED_ATTN=0
export VLLM_FL_ILUVATAR_PREFILL_BUCKETS=1
export VLLM_FL_ILUVATAR_LOGITS_GATHER=1
unset TIANSHU_PACK_PROBE_MODE VLLM_FL_ILUVATAR_HYBRID_STATS_PATH TIANSHU_BUCKET_RECHECK_TRACE TIANSHU_BUCKET_TRACE_DIR
export PYTHONPATH="${script_dir}/../tianshu_decode_launch_20261012/hooks"
export TIANSHU_DECODE_LAUNCH_PROBE=1
export TIANSHU_DECODE_LAUNCH_RECORD_DIR="${TIANSHU_DECODE_LAUNCH_RECORD_DIR:-/workspace/benchmark-results/decode_stage1_launch_records}"
export TIANSHU_FA2_TILE_PROBE=1
export TIANSHU_NATIVE_PREFILL_PROBE=1
export TIANSHU_NATIVE_PREFILL_RECORD_DIR="${TIANSHU_NATIVE_PREFILL_RECORD_DIR:-/workspace/benchmark-results/native_prefill_launch_records}"
export TIANSHU_FA2_TILE_RECORD_DIR="${TIANSHU_FA2_TILE_RECORD_DIR:-/workspace/benchmark-results/fa2_tile_launch_records}"

exec vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072 \
  --max-num-batched-tokens 2048
