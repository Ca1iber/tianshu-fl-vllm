#!/usr/bin/env bash
set -euo pipefail
base="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
check_only=0
if [[ "${1:-}" == --check ]]; then check_only=1;shift;fi
test "$#" -eq 0
manifest="$base/generated_source_bundle.json"
jq -e 'length==26 and all(to_entries[];(.key|startswith("/workspace/benchmark-results/")) and (.key|endswith(".py") or endswith("/SCREEN_GATE_32g.json")) and (.value.bundle_relative==("runtime_sources"+.key)))' "$manifest" >/dev/null
restored=0;checked=0
while IFS= read -r -d '' target && IFS= read -r -d '' relative && IFS= read -r -d '' expected; do
    bundled="$base/$relative";actual=$(sha256sum < "$bundled");actual=${actual%% *};test "$actual" = "$expected"
    if ! test -f "$target"; then
        if ((check_only));then printf 'Missing source: %s\n' "$target" >&2;exit 1;fi
        mkdir -p -- "$(dirname -- "$target")";cp -- "$bundled" "$target";restored=$((restored+1))
    fi
    actual=$(sha256sum < "$target");actual=${actual%% *};test "$actual" = "$expected"
    checked=$((checked+1))
done < <(jq -j 'to_entries[]|.key,"\u0000",.value.bundle_relative,"\u0000",.value.sha256,"\u0000"' "$manifest")
test "$checked" -eq 26
printf 'Generated sources checked: %s; restored: %s\n' "$checked" "$restored"
