#!/usr/bin/env bash
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
test "$#" -le 1
mode=${1:-restore};case "$mode" in restore|--check);;*)exit 2;;esac
sha256sum --check --status "$base/QUALIFIED_RELEASE.sha256"
mapfile -d '' -t evidence < <(jq -j '.entries[]|.sha256,"\u0000",.original_path,"\u0000",.gzip_file,"\u0000"' "$base/CONTROLLER_EVIDENCE_32g.json")
checked=0;restored=0;missing=0
for ((i=0;i<${#evidence[@]};i+=3));do
 expected=${evidence[i]};target=${evidence[i+1]};packed="$base/${evidence[i+2]}"
 case "$target" in /workspace/benchmark-results/20261022_native_int8_down_prefill_math8_v2_retry1_tianshu/*.json|/workspace/benchmark-results/20261022_native_int8_down_prefill_math64_v2_retry1_tianshu/*.json);;*)printf 'Unexpected evidence destination: %s\n' "$target" >&2;exit 1;;esac
 if [[ -f "$target" ]];then
  actual=$(sha256sum < "$target");actual=${actual%% *};test "$actual" = "$expected";checked=$((checked+1));continue
 fi
 if [[ "$mode" == --check ]];then printf 'Missing controller evidence: %s\n' "$target" >&2;missing=$((missing+1));continue;fi
 mkdir -p -- "$(dirname -- "$target")"
 temporary=$(mktemp "${target}.tianshu_restore.XXXXXX")
 gzip -cd -- "$packed" > "$temporary"
 actual=$(sha256sum < "$temporary");actual=${actual%% *}
 if [[ "$actual" != "$expected" ]];then rm -f -- "$temporary";printf 'Evidence hash mismatch: %s\n' "$target" >&2;exit 1;fi
 test ! -e "$target";mv -- "$temporary" "$target";restored=$((restored+1))
done
printf 'Controller evidence checked=%s restored=%s missing=%s\n' "$checked" "$restored" "$missing"
test "$missing" -eq 0
