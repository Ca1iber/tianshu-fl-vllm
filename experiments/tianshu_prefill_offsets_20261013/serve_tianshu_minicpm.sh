#!/usr/bin/env bash
set -euo pipefail

# Source-matched Prefill stage2 +Decode address32: fresh Math103/105.
# Official4K3213.13/16K2665.42, all requests successful; BF16, no speculation.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
record_dir="${TIANSHU_PREFILL_STAGE2_LAUNCH_RECORD_DIR:-/workspace/benchmark-results/prefill_stage2_launch_records}"
export VLLM_PLUGINS=fl
export VLLM_FL_ILUVATAR_HYBRID_ATTENTION=1
export FLAGGEMS_ILUVATAR_COMPACT_VARLEN=1
export VLLM_FL_ILUVATAR_ADAPTIVE_DECODE_SEGMENTS=1
export VLLM_FL_USE_FLAGGEMS_ATTN=0
export VLLM_FL_ILUVATAR_SPLIT_MIXED_ATTN=0
export VLLM_FL_ILUVATAR_PREFILL_BUCKETS=1
export VLLM_FL_ILUVATAR_LOGITS_GATHER=1
unset TIANSHU_PACK_PROBE_MODE VLLM_FL_ILUVATAR_HYBRID_STATS_PATH
unset TIANSHU_SORT_COUNT_BITS TIANSHU_SORT_COUNT_RECORD_DIR
unset TIANSHU_DECODE_GRAPH_MAX_ROWS TIANSHU_DECODE_GRAPH_RECORD_DIR
unset TIANSHU_SORT_BUFFER_MODE TIANSHU_SORT_BUFFER_RECORD_DIR
unset TIANSHU_SQL_REFLECTION_PROBE TIANSHU_SQL_REFLECTION_RECORD_DIR
export PYTHONPATH="${script_dir}/hooks_stage2"
export TIANSHU_DECODE_LAUNCH_PROBE=1
export TIANSHU_DECODE_LAUNCH_RECORD_DIR="${record_dir}/decode"
export TIANSHU_FA2_TILE_PROBE=1
export TIANSHU_NATIVE_PREFILL_PROBE=1
export TIANSHU_NATIVE_PREFILL_RECORD_DIR="${record_dir}/native_prefill"
export TIANSHU_FA2_TILE_RECORD_DIR="${record_dir}/fa2"
export TIANSHU_BUCKET_RECHECK_TRACE=1
export TIANSHU_BUCKET_TRACE_DIR="${record_dir}/buckets"
export TIANSHU_ADDRESS32_PROBE=1
export TIANSHU_ADDRESS32_RECORD_DIR="${record_dir}/address32"
export TIANSHU_PREFILL_STAGE2_PROBE=1
export TIANSHU_PREFILL_STAGE2_RECORD_DIR="${record_dir}/prefill_stage2"

exec vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072 \
  --max-num-batched-tokens 2048
