# SPDX-License-Identifier: Apache-2.0
"""Bind the actual shell launcher's dry run to completed Math and formal evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--math-root', type=Path, required=True)
    p.add_argument('--official-root', type=Path, required=True)
    a = p.parse_args()
    base = Path(__file__).resolve().parent
    repo = base.parents[1]
    template = read(base / 'serve_profile_template.json')
    mr, br = a.math_root.resolve(), a.official_root.resolve()
    ms, bs = read(mr / 'result.json'), read(br / 'result.json')
    q, c = read(mr / 'accuracy_audit.json'), read(br / 'comparison.json')
    for root in (mr, br):
        assert read(root / 'result.json')['phase'] == 'complete'
        assert read(root / 'local_validation_audit.json')['complete']
    assert q['total'] == 105 and q['accuracy'] > .95
    assert q['execution_summary']['errored'] == 0
    assert c['requests_successful'] == 1024 and c['excluded_runs'] == [1]
    assert c['passes_public_4k_performance_gates'] and c['profile']['budget'] == 4096
    assert template['frozen_sources'] == ms['frozen_sources'] == bs['frozen_sources']
    assert len(template['frozen_sources']) == 205
    assert ms['server_command'] == bs['server_command'] == template['server_command']
    target = '/tmp/tianshu_budget_default_launcher_audit_only'
    env = dict(os.environ)
    env.update(TIANSHU_K8V16_PROFILE_DRY_RUN='1',
        TIANSHU_K8V16_PROFILE_RECORD_DIR=target,
        TIANSHU_UNRELATED_STALE='BAD', FLAGGEMS_ILUVATAR_STALE='BAD',
        VLLM_FL_ILUVATAR_LOGITS_GATHER='BAD', VLLM_PLUGINS='BAD',
        PYTHONPATH='/tmp/not_an_import_path')
    dry = json.loads(subprocess.check_output(
        ['bash', str(base / 'serve_tianshu_minicpm.sh')], env=env, text=True))
    assert dry['command'] == template['server_command']
    for state in (ms, bs):
        expected = {k: v.replace(state['output_root'] + '/', target + '/')
            .replace('/workspace/vllm-plugin-FL/', str(repo) + '/')
            for k, v in state['profile_environment'].items()}
        assert len(expected) == 43
        assert dry['profile_environment'] == dry['selected_actual_environment'] == expected
    result = dict(complete=True, CPU_dry_run_passed=True,
        fresh_Math_pending=False, formal_comparison_pending=False,
        actual_bash_launcher_executed=True, stale_profile_environment_removed=True,
        selected_actual_environment_keys=43, source_snapshot_count=205,
        original_serve_argv=True, framework_default_budget=4096,
        launcher_sha256=digest(base / 'serve_tianshu_minicpm.sh'),
        template_sha256=digest(base / 'serve_profile_template.json'),
        fresh_Math_correct=q['correct'], fresh_Math_total=105,
        official_4k_total_tok_s=c['total_tok_s'],
        standalone_submission_reproduction_validated=False,
        launcher_verification_scope='Actual shell dry run compared with both completed APIs; no additional GPU launch')
    (base / 'LAUNCHER_AUDIT_32g.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
