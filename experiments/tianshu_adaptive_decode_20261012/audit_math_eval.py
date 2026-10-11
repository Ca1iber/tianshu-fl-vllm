# SPDX-License-Identifier: Apache-2.0
"""Audit independent EvalScope completion; --ignore-errors is not sufficient."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    state = json.loads((root / 'result.json').read_text())
    assert state['phase'] == 'evaluation_finished_pending_report_audit', state['phase']
    assert state['evalscope_exit_code'] == 0
    reports = list(root.glob('evalscope/*/reports/minicpm/math_500.json'))
    assert len(reports) == 1, reports
    report_path = reports[0]
    report = json.loads(report_path.read_text())
    execution = report['execution_summary']
    assert execution['requested'] == execution['succeeded'] == report['num'] == 105
    assert execution['errored'] == 0 and execution['incomplete'] is False
    metric = next(m for m in report['metrics'] if m['identity']['name'] == 'accuracy')
    assert metric['num'] == 105
    run = report_path.parents[2]
    review_path = run / 'reviews/minicpm/math_500_Level 3.jsonl'
    prediction_path = run / 'predictions/minicpm/math_500_Level 3.jsonl'
    reviews = [json.loads(line) for line in review_path.read_text().splitlines()]
    predictions = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    assert len(reviews) == len(predictions) == 105
    assert len({r['index'] for r in reviews}) == 105
    assert len({p['index'] for p in predictions}) == 105
    assert all(p['model_output']['error'] is None for p in predictions)
    stop_reasons = Counter(p['model_output']['choices'][0]['stop_reason'] for p in predictions)
    scores = [r['sample_score']['score'] for r in reviews]
    assert all(s['status'] == 'success' for s in scores)
    values = [s['value']['accuracy'] for s in scores]
    assert all(v in (0, 1) for v in values)
    correct = int(sum(values))
    assert abs(metric['score'] - correct / 105) < 0.000051
    hardware = (root / 'hardware_after.txt').read_text()
    assert 'No running processes found' in hardware, 'GPU cleanup requires investigation'
    audit = {
        'audited_at_utc': datetime.now(timezone.utc).isoformat(),
        'candidate': state['candidate'], 'complete': True,
        'adaptive_flag': state['adaptive_flag'], 'adaptive_sha256': state['adaptive_sha256'],
        'adaptive_source': state['adaptive_source'],
        'reused_original_accuracy': False, 'correct': correct, 'total': 105,
        'accuracy': correct / 105, 'report_accuracy': metric['score'],
        'official_requirement': 'accuracy > 0.95',
        'official_requirement_passed': correct / 105 > 0.95,
        'execution_summary': execution, 'review_statuses': {'success': 105},
        'incorrect_indices': [r['index'] for r, v in zip(reviews, values) if not v],
        'generation_stop_reasons': dict(stop_reasons),
        'max_tokens_indices': [p['index'] for p in predictions
                               if p['model_output']['choices'][0]['stop_reason'] == 'max_tokens'],
        'hybrid_source': state['hybrid_source'], 'hybrid_sha256': state['hybrid_sha256'],
        'evaluation_command': state['evaluation_command'],
        'gpu_idle_after': True,
        'raw_files': {str(p.relative_to(root)): {
            'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()
        } for p in (report_path, review_path, prediction_path)},
    }
    (root / 'accuracy_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
