#!/usr/bin/env bash
set -euo pipefail

# Reproduce the verified operator profile using the original official serve argv.
# Validation and formal score are documented separately; this script asserts no score.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
env -u PYTHONPATH python3 - "${script_dir}" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

base = Path(sys.argv[1]).resolve()
repo = base.parents[1]
sys.path.insert(0, str(base.parent / 'tianshu_original_command_20261016'))
from validation_profile import official_command

template = json.loads((base / 'serve_profile_template.json').read_text())
dry = os.environ.get('TIANSHU_K8V16_PROFILE_DRY_RUN') == '1'
root = Path(os.environ.get('TIANSHU_K8V16_PROFILE_RECORD_DIR', '/workspace/benchmark-results/k8v16_profile_' +
    datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + str(os.getpid()))).resolve()
old_root, old_repo = template['old_output_root'], template['old_repository_root']
profile = {key: value.replace(old_root + '/', str(root) + '/').replace(old_repo + '/', str(repo) + '/')
    for key, value in template['environment'].items()}
assert len(profile) == 41 and all(old_root not in value for value in profile.values())
assert profile['TIANSHU_TOPP_PRECISION_SERVICE_MODE'] == 'candidate'
assert profile['TIANSHU_PREFILL_GEMM_TILES_PROBE'] == '1'
assert profile['TIANSHU_MAX_SEQS_MODE'] == 'control'
assert profile['TIANSHU_SOFT8_SERVICE_MODE'] == 'candidate'
assert profile['TIANSHU_K8V16_SERVICE_MODE'] == 'candidate'
command = official_command()
assert command == template['server_command']
environment = dict(os.environ)
for key in list(environment):
    if key.startswith(('TIANSHU_', 'VLLM_FL_', 'FLAGGEMS_ILUVATAR_')) or key in {
        'VLLM_PLUGINS', 'PYTHONPATH', 'PYTHONUNBUFFERED', 'TRITON_PRINT_AUTOTUNING', 'CUDA_VISIBLE_DEVICES'
    }:
        del environment[key]
environment.update(profile)
if dry:
    selected_actual = {key: value for key, value in environment.items()
        if key in profile or key.startswith(('TIANSHU_', 'VLLM_FL_', 'FLAGGEMS_ILUVATAR_'))}
    print(json.dumps(dict(command=command, profile_environment=profile,
        selected_actual_environment=selected_actual, repository=str(repo),
        record_root=str(root), source_validation='required before real launch')))
else:
    for source, expected in template['frozen_sources'].items():
        path = Path(source.replace(old_repo + '/', str(repo) + '/'))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, str(path)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'launcher_profile.json').write_text(json.dumps(dict(command=command,
        profile_environment=profile, template_sha256=hashlib.sha256((base / 'serve_profile_template.json').read_bytes()).hexdigest(),
        all196_sources_verified=True), indent=2) + '\n')
    os.execvpe(command[0], command, environment)
PY
