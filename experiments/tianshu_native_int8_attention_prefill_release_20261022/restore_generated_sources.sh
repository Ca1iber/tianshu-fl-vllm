#!/usr/bin/env bash
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
exec bash "$base/../tianshu_m_range_prefill_release_20261021/restore_generated_sources.sh" "$@"
