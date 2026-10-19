# SPDX-License-Identifier: Apache-2.0
"""Compare an actual shell dry run with independently completed validation APIs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

base = Path(__file__).resolve().parent
repo = base.parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--math-root', type=Path, required=True)
parser.add_argument('--official-root', type=Path, required=True)
args = parser.parse_args()

def read(path):
    return json.loads(path.read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

math_root, official_root = args.math_root.resolve(), args.official_root.resolve()
states = [read(root / 'result.json') for root in (math_root, official_root)]
assert all(s['phase'] == 'complete' and not s['reused_original_accuracy'] for s in states)
assert states[0]['stage'] == 'math' and states[0]['total'] == 105 and states[0]['accuracy'] > .95
assert states[1]['stage'] == 'official' and states[1]['completed_runs'] == 4
assert states[0]['frozen_sources'] == states[1]['frozen_sources']
template = read(base / 'serve_profile_template.json')
assert template['frozen_sources'] == states[0]['frozen_sources']

with tempfile.TemporaryDirectory(prefix='tianshu-k8v16-launcher-') as scratch:
    environment = dict(os.environ)
    environment.update(TIANSHU_K8V16_PROFILE_DRY_RUN='1',
        TIANSHU_K8V16_PROFILE_RECORD_DIR=scratch,
        TIANSHU_STALE_AUDIT_SENTINEL='must_be_removed',
        VLLM_FL_STALE_AUDIT_SENTINEL='must_be_removed')
    actual = json.loads(subprocess.check_output(
        ['bash', str(base / 'serve_tianshu_minicpm.sh')], env=environment, text=True))
    assert actual['selected_actual_environment'] == actual['profile_environment']
    assert len(actual['profile_environment']) == 41
    comparisons = []
    for root, state in zip((math_root, official_root), states):
        observed = read(root / 'observed_server_environment.json')
        assert observed['environment'] == state['profile_environment']
        expected = {k: v.replace(state['output_root'] + '/', scratch + '/').replace(
            '/workspace/vllm-plugin-FL/', str(repo) + '/')
            for k, v in state['profile_environment'].items()}
        assert actual['command'] == state['server_command'] == observed['command']
        assert actual['profile_environment'] == expected
        comparisons.append(dict(stage=state['stage'], selected_actual_API_environment_keys=41,
            original_serve_argv=True, actual_shell_dry_run_matches_API=True))

audit = dict(complete=True, actual_shell_dry_run=True, comparisons=comparisons,
    source_snapshot_count=len(states[0]['frozen_sources']),
    selected_actual_environment_has_no_extra_switches=True,
    inherited_stale_switches_removed=True,
    actual_live_launcher_not_run=True,
    launcher_sha256=digest(base / 'serve_tianshu_minicpm.sh'),
    template_sha256=digest(base / 'serve_profile_template.json'),
    formal_comparison_pending=False,
    scope='Actual shell dry run compared with both completed APIs; no extra live launch via launcher')
(base / 'LAUNCHER_AUDIT_32g.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit))
