"""Evidence for resuming the SAME candidate; cached answers never change."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import threading
import time
import urllib.request

from prepare_resume_cache import EVALSCOPE_SOURCES, FINGERPRINT, PREDICTIONS_SHA


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_cache(root, seed):
    seed = Path(seed).resolve()
    manifest = json.loads((seed / 'resume_manifest.json').read_text())
    assert manifest['kind'] == 'prepared_incomplete_cache' and not manifest['complete_quality']
    assert manifest['predictions_preserved_verbatim'] and manifest['all_wrong_answers_retained']
    assert manifest['cached_prediction_count'] == 103 and manifest['missing_indices'] == [83, 101]
    assert manifest['evaluation_fingerprint'] == FINGERPRINT
    assert all(digest(seed / p) == sha for p, sha in manifest['files_sha256'].items())
    preflight = Path('/tmp/tianshu_math_resume_real_cache_preflight_20261016.json')
    checked = json.loads(preflight.read_text())
    assert checked['actual_installed_cache_filter_passed'] and checked['all103_deserialized_against_real105_dataset']
    assert checked['remaining_indices'] == [83, 101] and checked['unchanged_evaluation_fingerprint'] == FINGERPRINT
    cache = root / 'evalscope/20261016_resumed'
    assert not cache.exists()
    shutil.copytree(seed, cache)
    assert not (cache / 'reviews').exists() and not (cache / 'reports').exists()
    extra = {str(Path(__file__)): digest(__file__)}
    for name in ('run_math_resume.py', 'prepare_resume_cache.py', 'verify_incomplete_downloaded.py', 'probe_installed_cache.py'):
        path = Path(__file__).with_name(name); extra[str(path)] = digest(path)
    package = Path('/usr/local/lib/python3.12/site-packages/evalscope')
    for name, sha in EVALSCOPE_SOURCES.items():
        path = package / name; assert digest(path) == sha; extra[str(path)] = sha
    extra[str(package / 'run.py')] = digest(package / 'run.py')
    assert len(extra) == 10
    assert not checked['fingerprint_override_required']
    assert checked['probe_source_SHA256'] == extra[str(Path(__file__).with_name('probe_installed_cache.py'))]
    for name, sha in extra.items():
        target = root / 'orchestration_source_snapshot' / Path(name).relative_to('/')
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(Path(name).read_bytes())
        assert digest(target) == sha
    (root / 'installed_cache_preflight.json').write_bytes(preflight.read_bytes())
    return cache, extra, manifest


def canary(root):
    """Exercise real B8 TopP; all responses saved, separate from Math105."""
    barrier = threading.Barrier(8)
    def one(index):
        payload = dict(model='minicpm', messages=[dict(role='user', content=
            f'Probe {index}. Continue listing consecutive integers after 1000. Use spaces.')],
            max_tokens=64, ignore_eos=True, temperature=1.0, top_p=.95, seed=2026101664+index)
        request = urllib.request.Request('http://127.0.0.1:9031/v1/chat/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        barrier.wait(timeout=10); started = time.time()
        with urllib.request.urlopen(request, timeout=120) as response:
            assert response.status == 200; body = json.load(response)
        assert body['usage']['completion_tokens'] == 64 and len(body['choices']) == 1
        return dict(index=index, payload=payload, response=body, started=started, finished=time.time())
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(one, range(8)))
    (root / 'profile_canary.json').write_text(json.dumps(dict(kind='profile_route_probe_not_Math',
        count=8, output_tokens_each=64, rows=rows), indent=2) + '\n')
    routes = [json.loads(p.read_text()) for p in (root / 'topp_launch_records').glob('route_*.json')]
    assert any(r['batch'] == 8 and r['topp'] and not r['topk'] and r['corrected_source_active']
        and r['actual_source_sha256'] == '09156e66b6aa2f4460423ea5b8d70bee043022bd75f0a784417455d0e979479d'
        for r in routes), 'B8 candidate must execute before two-question resume'


def audit_completed(root, state, online=True, write=True):
    cache = root / 'evalscope/20261016_resumed'
    manifest = json.loads((cache / 'resume_manifest.json').read_text())
    assert state['resumed_same_candidate_predictions'] and state['cached_prediction_count'] == 103
    assert state['new_prediction_indices'] == [83, 101]
    assert manifest['frozen_sources'] == state['frozen_sources']
    for name, sha in state['supplemental_sources'].items():
        assert digest(root / 'orchestration_source_snapshot' / Path(name).relative_to('/')) == sha
        if online: assert digest(name) == sha
    path = cache / 'predictions/minicpm/math_500_Level 3.jsonl'
    lines = path.read_bytes().splitlines(keepends=True)
    assert len(lines) == 105 and hashlib.sha256(b''.join(lines[:103])).hexdigest() == PREDICTIONS_SHA
    rows = [json.loads(line) for line in lines]
    assert {r['index'] for r in rows} == set(range(105))
    assert sorted(r['index'] for r in rows[103:]) == [83, 101]
    assert {str(r['index']): hashlib.sha256(line.rstrip(b'\r\n')).hexdigest()
        for r, line in zip(rows[:103], lines[:103])} == manifest['cached_prediction_line_SHA256']
    # The regenerated task config contains this exact benchmark fingerprint.
    config = (cache / 'configs/task_config.yaml').read_text()
    assert FINGERPRINT in config and 'rerun_review: true' in config
    reviews = [json.loads(line) for line in (cache / 'reviews/minicpm/math_500_Level 3.jsonl').read_text().splitlines()]
    assert len(reviews) == 105 and {r['index'] for r in reviews} == set(range(105))
    assert all(r['sample_score']['score']['status'] == 'success' for r in reviews)
    count = lambda name: sum(float(value) for value in re.findall(
        r'^vllm:request_success_total(?:\{[^}]*\})? ([0-9.eE+\-]+)$',
        (root / name).read_text(), re.MULTILINE))
    assert count('after_metrics.txt') - count('before_metrics.txt') == 2
    probe = json.loads((root / 'profile_canary.json').read_text())
    assert probe['count'] == len(probe['rows']) == 8
    assert all(r['response']['usage']['completion_tokens'] == 64 for r in probe['rows'])
    result = dict(complete=True, same_candidate_resume=True, all103_cached_prediction_bytes_unchanged=True,
        all_wrong_answers_retained=True, new_prediction_indices=[83, 101], new_API_requests=2,
        all105_predictions_unique=True, all105_reviews_regenerated=True,
        original_candidate_data_fingerprint_preserved=True, original_target58_sources_preserved=True,
        supplemental_orchestration_and_EvalScope_sources_verified=len(state['supplemental_sources']),
        original_accuracy_not_reused=True, profile_probe_requests_excluded_from_Math=8)
    if write: (root / 'resume_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
