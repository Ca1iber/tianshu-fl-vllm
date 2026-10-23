#!/usr/bin/env bash
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
exec bash "$base/../tianshu_native_int8_attention_prefill_release_20261022/restore_generated_sources.sh" "$@"
