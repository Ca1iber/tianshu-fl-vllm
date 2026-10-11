# SPDX-License-Identifier: Apache-2.0
"""Bounded continuous-refill control/candidate pair, never an official score."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-batched-tokens', choices=[2048, 4096, 8192], type=int, default=8192)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    trial = Path(__file__).with_name('run_service_trial.py')
    for gather in ('0', '1'):
        output = root / f'gather_{gather}'
        output.mkdir(exist_ok=True)
        if (output / 'result.json').exists():
            raise RuntimeError('Refusing to overwrite an existing refill trial')
        command = [sys.executable, '-u', str(trial), '--output', str(output),
                   '--buckets', '0', '--gather', gather, '--hybrid', '1', '--adaptive', '1',
                   '--max-batched-tokens', str(args.max_batched_tokens),
                   '--requests', '192', '--concurrency', '64', '--output-tokens', '128',
                   '--rounds', '3']
        with (output / 'runner.log').open('w') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        result = json.loads((output / 'result.json').read_text())
        print(json.dumps({'gather': gather, 'phase': result['phase'],
                          'returncode': process.returncode,
                          'stability': result.get('measured_round_stability')},
                         ensure_ascii=False), flush=True)
        if process.returncode or result['phase'] != 'complete':
            raise RuntimeError('Preserve the failed refill trial and inspect it')
        if 'No running processes found' not in subprocess.check_output(['ixsmi'], text=True):
            raise RuntimeError('GPU must be idle after owned service cleanup')


if __name__ == '__main__':
    main()
