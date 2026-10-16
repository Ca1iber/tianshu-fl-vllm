"""Copy this candidate's 103 predictions into a new, incomplete EvalScope cache.

No API calls, evaluation or score promotion. Regrade all answers on future use.
The original snapshot is read only. Installed EvalScope integration remains a
separate prerequisite before any GPU run or claim of full 105-question accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from verify_incomplete_downloaded import verify

CONFIG_SHA = 'dbf56a0bb8a092bf5fbba7de862d43258ac087cb0f78b39bd2b8c6283d2f02d5'
SNAPSHOT_SHA = 'd8948ca76a6c783ef2294fe245ff4905f422383c798964a20fca41025918d65e'
FINGERPRINT = 'sha256:d2a0a846c4e8e0b074d9cf260e5bb5c3a1d6e41086e4ecefc7d17c624949151a'
PREDICTIONS_SHA = 'b3aef70457201b723b0d09cf69bfed3995fe0ac3ddc2aa32c03d45f04697c2ce'
EVALSCOPE_SOURCES = {
    'arguments.py': 'daedad68adce7e2d047d0ca720ec1424dd8a5e3687e2e3ad5925a9fd4ecfbfa9',
    'config.py': 'ecd0ed8be4485213ce67eaf0a6047d72b4a669d17b51014cc32eef425b1640e9',
    'evaluator/evaluator.py': 'a7c1934ea24effe91ef1622fdf4df823055b76f5dcd246356cebc4a29236081a',
    'api/evaluator/cache.py': 'f0d127bc00898aba93efe404f1574bcc32ab04757824984425d24eb9346184d3',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(snapshot, cache_sources, output):
    snapshot, cache_sources, output = (Path(p).resolve() for p in (snapshot, cache_sources, output))
    assert not output.exists() and not output.is_relative_to(snapshot) and not output.is_relative_to(cache_sources)
    audit = verify(snapshot)
    assert not audit['complete_quality']
    meta_bytes = (snapshot / 'TERMINAL_SNAPSHOT.json').read_bytes()
    assert digest(meta_bytes) == SNAPSHOT_SHA
    state = json.loads((snapshot / 'result.json').read_text())
    observed = json.loads((snapshot / 'observed_server_environment.json').read_text())
    assert observed['environment'] == state['profile_environment'] and len(observed['environment']) == 38
    assert observed['command'] == state['server_command']
    configs = list(snapshot.glob('evalscope/*/configs/task_config.yaml'))
    predictions = list(snapshot.glob('evalscope/*/predictions/minicpm/*.jsonl'))
    reviews = list(snapshot.glob('evalscope/*/reviews/minicpm/*'))
    assert len(configs) == len(predictions) == len(reviews) == 1
    config, predicted, reviewed = (paths[0].read_bytes() for paths in (configs, predictions, reviews))
    assert digest(config) == CONFIG_SHA and FINGERPRINT.encode() in config
    assert digest(predicted) == PREDICTIONS_SHA and predicted.endswith(b'\n')
    p_rows = [json.loads(line) for line in predicted.splitlines()]
    r_rows = {row['index']: row for row in map(json.loads, reviewed.splitlines())}
    row_hashes = {}
    for row, line in zip(p_rows, predicted.splitlines()):
        index = row['index']; model_output = row['model_output']; review = r_rows[index]
        assert row['model'] == model_output['model'] == 'minicpm'
        assert model_output['error'] is None and len(model_output['choices']) == 1
        assert model_output['choices'][0]['stop_reason'] in ('stop', 'max_tokens')
        assert row['messages'] == review['messages']
        assert row['messages'][-1] == model_output['choices'][0]['message']
        content = row['messages'][-1]['content']
        if isinstance(content, list):
            assert all(part['type'] in ('reasoning', 'text') for part in content)
            visible_text = ''.join(part['text'] for part in content if part['type'] == 'text')
        else:
            assert isinstance(content, str)
            visible_text = content
        assert review['sample_score']['score']['prediction'] == visible_text
        assert review['sample_score']['score']['value']['accuracy'] in (0, 1)
        row_hashes[str(index)] = digest(line)
    assert len(row_hashes) == 103 and set(map(int, row_hashes)) == set(range(105)) - {83, 101}
    source_manifest = json.loads((cache_sources / 'source_manifest.json').read_text())
    assert source_manifest['files'] == EVALSCOPE_SOURCES
    source_bytes = {name: (cache_sources / name).read_bytes() for name in source_manifest['files']}
    assert len(source_bytes) == 4
    assert all(digest(data) == source_manifest['files'][name] for name, data in source_bytes.items())
    files = {
        'configs/task_config.yaml': config,
        'predictions/minicpm/math_500_Level 3.jsonl': predicted,
        'provenance/original_transactional_reviews.jsonl': reviewed,
        'provenance/original_result.json': (snapshot / 'result.json').read_bytes(),
        'provenance/original_terminal_snapshot.json': meta_bytes,
        'provenance/observed_server_environment.json': (snapshot / 'observed_server_environment.json').read_bytes(),
    }
    for name, data in source_bytes.items():
        files['provenance/evalscope/' + name] = data
    manifest = dict(kind='prepared_incomplete_cache', complete_quality=False, evaluation_started=False,
        installed_EvalScope_integration_verified=False, source_snapshot=str(snapshot),
        source_archive_sha256=audit['archive_sha256'], evaluation_fingerprint=FINGERPRINT,
        predictions_preserved_verbatim=True, cached_prediction_count=103, missing_indices=[83, 101],
        cached_prediction_line_SHA256=row_hashes, all_wrong_answers_retained=True,
        reviews_for_provenance_only=True, rerun_review_required=True, old_accuracy_not_reused=True,
        source_config_sha256=CONFIG_SHA, frozen_sources=state['frozen_sources'],
        server_command=state['server_command'], profile_environment=state['profile_environment'],
        original_evaluation_command=state['evaluation_command'],
        files_sha256={name: digest(data) for name, data in files.items()},
        prerequisites=['New reserved slot and same target/profile source hashes',
            'Installed EvalScope validates 103 cached rows against the full105 dataset and original fingerprint',
            'Use a disposable working copy with --use-cache plus --rerun-review and unchanged sampling',
            'Final105 unique successful predictions/reviews,103 original prediction rows unchanged; no old accuracy'],
        full_accuracy_not_claimed=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.math_cache_prepare_', dir=output.parent) as temporary:
        staged = Path(temporary) / 'cache'; staged.mkdir()
        for name, data in files.items():
            target = staged / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        (staged / 'resume_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        assert not (staged / 'reports').exists() and not (staged / 'reviews').exists()
        assert all(digest((staged / name).read_bytes()) == sha for name, sha in manifest['files_sha256'].items())
        staged.rename(output)
    assert (snapshot / 'TERMINAL_SNAPSHOT.json').read_bytes() == meta_bytes
    assert digest(predictions[0].read_bytes()) == PREDICTIONS_SHA
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot-root', type=Path, required=True)
    parser.add_argument('--cache-sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.snapshot_root, args.cache_sources, args.output)
    print(json.dumps({key: result[key] for key in ('kind', 'complete_quality', 'cached_prediction_count',
        'missing_indices', 'predictions_preserved_verbatim', 'installed_EvalScope_integration_verified')}))


if __name__ == '__main__':
    main()
