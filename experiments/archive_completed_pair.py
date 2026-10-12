# SPDX-License-Identifier: Apache-2.0
"""Freeze an owned, completed experiment and its source bundle in checked parts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--single-probe', action='store_true')
    parser.add_argument('--completion-file', choices=['result.json','accuracy_audit.json'], default='result.json')
    args = parser.parse_args()
    root = args.source.resolve()
    assert root.is_relative_to('/workspace/benchmark-results') and root.is_dir()
    assert args.label.replace('_', '').isalnum()
    comparison = root / (args.completion_file if args.single_probe else 'comparison.json')
    deadline = datetime.fromisoformat(json.loads(Path(
        '/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json'
    ).read_text())['stop_at_utc'])
    def completed():
        if not comparison.exists():
            return False
        if not args.single_probe:
            return True
        result = json.loads(comparison.read_text())
        return result.get('complete') is True or result.get('phase') == 'complete'
    while not completed():
        if not args.wait or datetime.now(timezone.utc) >= deadline:
            raise RuntimeError('No completed comparison to archive')
        time.sleep(5)
    states = [comparison] if args.single_probe else list(root.glob('*/result.json'))
    if not args.single_probe:
        assert len(states) == 2 and all(json.loads(p.read_text())['phase'] == 'complete' for p in states)
    repository = Path('/workspace/vllm-plugin-FL')
    commit = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
    bundle = root / 'source_repository.bundle'
    subprocess.run(['git', '-C', str(repository), 'bundle', 'create', str(bundle), 'HEAD'], check=True, capture_output=True)
    (root / 'archive_provenance.json').write_text(json.dumps({
        'created_utc': datetime.now(timezone.utc).isoformat(), 'repository_commit': commit,
        'repository_dirty_status': subprocess.check_output(['git', '-C', str(repository), 'status', '--porcelain'], text=True),
        'completed_results': [str(p.relative_to(root)) for p in states],
        'all_existing_raw_files_included': True,
        'raw_response_scope': 'All response files produced by the original experiment; no missing response bodies are reconstructed'}, indent=2) + '\n')
    archive = Path('/tmp') / f'tianshu_{args.label}.tgz'
    assert not archive.exists(), 'Archive names are immutable'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(root, arcname=root.name)
    data = archive.read_bytes()
    parts = []
    for index, offset in enumerate(range(0, len(data), 262144)):
        part = archive.with_name(archive.name + f'.part{index:03d}')
        chunk = data[offset:offset + 262144]
        part.write_bytes(chunk)
        parts.append({'name': part.name, 'bytes': len(chunk), 'sha256': hashlib.sha256(chunk).hexdigest()})
    manifest = {'archive': str(archive), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'parts': parts}
    output = Path('/tmp') / f'tianshu_{args.label}_manifest.json'
    output.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'manifest': str(output), 'bytes': len(data), 'sha256': manifest['sha256']}))


if __name__ == '__main__':
    main()
