# SPDX-License-Identifier: Apache-2.0
"""Verify both downloaded arms and recompute the matched service comparison."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    for mode in ('control', 'candidate'):
        subprocess.run([sys.executable, str(Path(__file__).with_name('verify_trial.py')),
                        '--root', str(root / mode)], check=True, stdout=subprocess.DEVNULL)
    read = lambda p: json.loads(p.read_text())
    control, candidate = (read(root / mode / 'result.json') for mode in ('control', 'candidate'))
    comparison = read(root / 'comparison.json')
    assert comparison['complete'] and comparison['all384_screen_requests_successful']
    assert control['server_command'] == candidate['server_command']
    assert control['frozen_sources'] == candidate['frozen_sources'] == comparison['frozen_sources']
    for name, value in control['profile_environment'].items():
        other = candidate['profile_environment'][name]
        if name == 'TIANSHU_RAGGED_SERVICE_MODE':
            assert (value, other) == ('control', 'candidate')
        else:
            assert value.replace(control['output_root'], '<RUN>') == other.replace(candidate['output_root'], '<RUN>')
    assert control['seeded_canaries'][0]['text'] == candidate['seeded_canaries'][0]['text']
    for mode, state in (('control', control), ('candidate', candidate)):
        assert comparison[mode + '_total_tok_s'] == state['mean_total_tok_s']
    gain = 100 * (candidate['mean_total_tok_s'] / control['mean_total_tok_s'] - 1)
    assert comparison['gain_percent'] == gain
    result = dict(complete=True, all384_requests_and10_canaries=True,
                  control_total_tok_s=control['mean_total_tok_s'],
                  candidate_total_tok_s=candidate['mean_total_tok_s'], gain_percent=gain,
                  paired_configuration_and_sources_verified=True,
                  actual_mixed_graph_coverage_verified=True, request_count_padding=False,
                  fresh_Math_not_run=True, official_benchmark=False, adopted=False)
    (root / 'local_download_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
