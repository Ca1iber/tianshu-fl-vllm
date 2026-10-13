# SPDX-License-Identifier: Apache-2.0
"""Use the verified workload with separately archived in-process diagnostics."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    root.mkdir(parents=True)
    trial = Path(__file__).resolve().parents[1] / 'tianshu_native_prefill_20261012/run_service_trial.py'
    hook = Path(__file__).resolve().parent / 'hooks/sitecustomize.py'
    (root / 'run_diagnostic.py').write_bytes(Path(__file__).read_bytes())
    (root / 'diagnostic_sitecustomize.py').write_bytes(hook.read_bytes())
    (root / 'run_service_trial.py').write_bytes(trial.read_bytes())
    popen = subprocess.Popen

    def instrumented_popen(command, *positional, **kwargs):
        if isinstance(command, list) and command[:2] == ['vllm', 'serve']:
            env = dict(kwargs['env'])
            env['PYTHONPATH'] = str(hook.parent)
            env['TIANSHU_CPU_STACK_DIR'] = str(root / 'stacks')
            kwargs['env'] = env
        return popen(command, *positional, **kwargs)

    spec = importlib.util.spec_from_file_location('_tianshu_cold_diagnostic_trial', trial)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    subprocess.Popen = instrumented_popen
    sys.argv = [str(trial), '--output', str(root / 'candidate'),
                '--buckets', '1', '--gather', '1', '--bucket-trace', '2',
                '--hybrid', '1', '--adaptive', '1', '--tile', '1', '--native-prefill', '1',
                '--decode-launch', '1', '--max-batched-tokens', '2048',
                '--hist-count-bits', 'off', '--requests', '64', '--rounds', '2']
    try:
        code = module.main()
    finally:
        subprocess.Popen = popen
    state = json.loads((root / 'candidate/result.json').read_text())
    result = dict(phase=state['phase'], official_benchmark=False, adopted=False,
                  performance_score_allowed=False, diagnostic_only=True,
                  stack_files=[str(p.relative_to(root)) for p in (root / 'stacks').glob('*.jsonl')])
    (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    if code == 0:
        assert result['stack_files']
        subprocess.run(['python3', str(Path(__file__).resolve().parents[1] / 'archive_completed_pair.py'),
                        '--single-probe', '--source', str(root), '--label', 'cold_start_python_stacks_20261013'], check=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
