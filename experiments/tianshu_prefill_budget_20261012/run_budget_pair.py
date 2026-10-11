# SPDX-License-Identifier: Apache-2.0
"""Follow a promising short sweep with a serialized64-request4K/1K pair."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-budget', type=int, choices=[4096, 8192], required=True)
    parser.add_argument('--adaptive', choices=['0', '1'], default='1')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    trial = Path(__file__).with_name('run_budget_trial.py')
    for budget in (2048, args.candidate_budget):
        directory = root / f'budget_{budget}'
        directory.mkdir(exist_ok=True)
        if (directory / 'result.json').exists():
            raise RuntimeError('Refusing to overwrite a completed pair')
        command = [sys.executable, '-u', str(trial), '--output', str(directory),
                   '--buckets', '0', '--gather', '0', '--hybrid', '1',
                   '--adaptive', args.adaptive, '--max-batched-tokens', str(budget),
                   '--output-tokens', '1024', '--concurrency', '64', '--requests', '64',
                   '--rounds', '3']
        with (directory / 'runner.log').open('w') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        result = json.loads((directory / 'result.json').read_text())
        print(json.dumps({'budget': budget, 'phase': result['phase'],
                          'returncode': process.returncode,
                          'stability': result.get('measured_round_stability')},
                         ensure_ascii=False), flush=True)
        if process.returncode or result['phase'] != 'complete':
            raise RuntimeError(f'Budget{budget} failed; preserve logs')
        if 'No running processes found' not in subprocess.check_output(['ixsmi'], text=True):
            raise RuntimeError('GPU must be idle after owned service cleanup')
    subprocess.run([sys.executable, str(trial.with_name('summarize.py')),
                    '--output', str(root), '--budgets', '2048', str(args.candidate_budget),
                    '--expected-rounds', '3'], check=True)


if __name__ == '__main__':
    main()
