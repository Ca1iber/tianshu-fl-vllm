#!/usr/bin/env bash
set -euo pipefail
base="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
check_only=0
if [[ "${1:-}" == --check ]]; then check_only=1; shift; fi
test "$#" -eq 0
manifest="$base/generated_source_bundle.json"
jq -e 'length==22 and all(to_entries[];(.key|startswith("/workspace/benchmark-results/")) and (.key|endswith(".py")) and (.value.bundle_relative|startswith("runtime_sources/workspace/benchmark-results/")))' "$manifest" >/dev/null
restored=0;checked=0
while IFS= read -r -d '' target && IFS= read -r -d '' relative && IFS= read -r -d '' expected; do
  bundled="$base/$relative"
  actual=$(sha256sum < "$bundled"); actual=${actual%% *}; test "$actual" = "$expected"
  if ! test -f "$target"; then
    if (( check_only )); then printf 'Missing generated source: %s\n' "$target" >&2; exit 1; fi
    mkdir -p -- "$(dirname -- "$target")"
    cp -- "$bundled" "$target"
    restored=$((restored+1))
  fi
  actual=$(sha256sum < "$target"); actual=${actual%% *}
  if [[ "$actual" != "$expected" ]]; then printf 'Source SHA mismatch: %s\n' "$target" >&2; exit 1; fi
  checked=$((checked+1))
done < <(jq -j 'to_entries[]|.key,"\u0000",.value.bundle_relative,"\u0000",.value.sha256,"\u0000"' "$manifest")
test "$checked" -eq 22
printf 'Generated Python sources checked: %s; restored: %s\n' "$checked" "$restored"
