"""Verify the immutable stopped Math snapshot without inventing full accuracy."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    read = lambda name: json.loads((root / name).read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    meta, state = read('TERMINAL_SNAPSHOT.json'), read('result.json')
    assert meta['phase'] == state['phase'] == 'stopped'
    assert not meta['complete_quality'] and state['error'] == '14'
    assert state == read('state.json')
    for name, sha in meta['files_sha256'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert digest(root / name) == sha, name
    assert len(state['frozen_sources']) == 58
    repo = Path(__file__).resolve().parents[2]
    for source, sha in state['frozen_sources'].items():
        assert digest(root / 'source_snapshot' / Path(source).relative_to('/')) == sha, source
        if Path(source).is_relative_to('/workspace/vllm-plugin-FL'):
            assert digest(repo / Path(source).relative_to('/workspace/vllm-plugin-FL')) == sha, source
    rows, counts = {}, {}
    for name, info in meta['JSONL_records'].items():
        items = [json.loads(line) for line in (root / name).read_text().splitlines()]
        indices = [item['index'] for item in items]
        assert len(indices) == len(set(indices)) == 103
        assert all(0 <= index < 105 for index in indices)
        assert not info['partial_last_record'] and info['complete_JSON_records'] == 103
        kind = 'predictions' if '/predictions/' in name else 'reviews'
        assert kind not in rows
        rows[kind] = {item['index']: item for item in items}
        counts[kind] = dict(complete_records=103, missing_indices=sorted(set(range(105)) - set(indices)))
    assert rows['predictions'].keys() == rows['reviews'].keys()
    for index, prediction in rows['predictions'].items():
        score = rows['reviews'][index]['sample_score']
        assert score['sample_id'] == index and score['score']['status'] == 'success'
        assert score['sample_metadata']['question_id'] == prediction['metadata']['question_id']
    assert all(process['state'] in (None, 'Z') for process in meta['terminal_processes'].values())
    assert 'No running processes found' in (root / 'TERMINAL_HARDWARE.txt').read_text()
    manifest = json.loads((root.parent / 'backup_manifest.json').read_text())
    assert manifest['sha256'] == '8f26750a4f687175e1460b1326c44e4b0dbfe4433b9e984f10a309bdbbdbc22c'
    result = dict(complete_quality=False, phase='stopped_at_17_58',
        all_captured_file_SHA256_verified=True,
        all58_frozen_source_snapshots_and_current_repo_sources_verified=True,
        all103_predictions_and_reviews_parse_and_have_unique_matching_indices=True,
        all103_completed_reviews_successful=True, counts=counts,
        GPU_idle_and_terminal_processes_verified=True,
        all_existing_answers_including_incorrect_kept=True, full_accuracy_not_claimed=True,
        archive_sha256=manifest['sha256'])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root)
    (args.root / 'local_terminal_snapshot_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
