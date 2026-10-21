#!/usr/bin/env bash
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# The same26 frozen generated inputs are already archived with the qualified INT8 release.
exec bash "$base/../tianshu_prefill_int8_release_20261021/restore_generated_sources.sh" "$@"
