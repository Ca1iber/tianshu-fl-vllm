"""Validate real EvalScope/dataset/cache compatibility without model API calls."""
import argparse
import hashlib
import json
from pathlib import Path
import signal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); signal.alarm(90)
    from evalscope.config import TaskConfig, load_task_config_snapshot
    from evalscope.api.registry import get_benchmark
    from evalscope.api.evaluator.cache import CacheManager, ModelResult, ReviewResult
    from evalscope.utils.io_utils import OutputsStructure
    from evalscope.run import build_evaluation_identity, validate_cached_evaluation_identity, ResolvedBenchmarkSpec
    from prepare_resume_cache import FINGERPRINT, PREDICTIONS_SHA, CONFIG_SHA, EVALSCOPE_SOURCES

    cache = args.cache.resolve()
    hashes = {str(p.relative_to(cache)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in cache.rglob('*') if p.is_file()}
    assert hashes['configs/task_config.yaml'] == CONFIG_SHA
    assert hashes['predictions/minicpm/math_500_Level 3.jsonl'] == PREDICTIONS_SHA
    package = Path('/usr/local/lib/python3.12/site-packages/evalscope')
    assert all(hashlib.sha256((package / p).read_bytes()).hexdigest() == sha for p, sha in EVALSCOPE_SOURCES.items())
    cfg = TaskConfig(model='minicpm', api_url='http://127.0.0.1:9031/v1/chat/completions', api_key='EMPTY',
        eval_type='openai_api', datasets=['math_500'], dataset_args={'math_500': {
        'dataset_id': '/workspace/evalscope-datasets/math_500', 'subset_list': ['Level 3']}},
        eval_batch_size=8, timeout=3600, generation_config={'temperature': 1.0, 'top_p': .95, 'max_tokens': 32768},
        ignore_errors=True, use_cache=str(cache), rerun_review=True, work_dir=str(cache))
    benchmark = get_benchmark('math_500', cfg)
    # DatasetDict is deliberately accessed via items(), as DefaultEvaluator does.
    datasets = {key: value for key, value in benchmark.load_dataset().items()}
    assert set(datasets) == {'Level 3'} and len(datasets['Level 3']) == 105
    data = datasets['Level 3']
    original_ids = {i: data[i].metadata['question_id'] for i in range(105)}
    identity = build_evaluation_identity({'math_500': ResolvedBenchmarkSpec.from_meta(benchmark.benchmark_meta, cfg)},
        {'math_500': benchmark.benchmark_meta.evaluation_version}, cfg)
    original = load_task_config_snapshot(str(cache / 'configs/task_config.yaml'))
    assert identity.benchmarks['math_500'].fingerprint == FINGERPRINT
    cache_sources = validate_cached_evaluation_identity(original, identity, True)
    rows = [json.loads(line) for line in (cache / 'predictions/minicpm/math_500_Level 3.jsonl').read_text().splitlines()]
    assert all(ModelResult.model_validate(row).index == row['index'] for row in rows)
    assert all(row['metadata']['question_id'] == original_ids[row['index']] for row in rows)
    review_rows = [json.loads(line) for line in (cache / 'provenance/original_transactional_reviews.jsonl').read_text().splitlines()]
    assert len(review_rows) == 103
    assert all(ReviewResult.from_cache_item(row).index == row['index'] for row in review_rows)
    manager = CacheManager(OutputsStructure(outputs_dir=str(cache), is_make=False), 'minicpm', 'math_500')
    states, remaining = manager.filter_prediction_cache('Level 3', data)
    assert len(states) == 103 and len(remaining) == 2
    remaining_ids = sorted(remaining[i].id for i in range(len(remaining)))
    assert remaining_ids == [83, 101]
    assert hashes == {str(p.relative_to(cache)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in cache.rglob('*') if p.is_file()}
    result = dict(actual_installed_cache_filter_passed=True, all103_deserialized_against_real105_dataset=True,
        remaining_indices=remaining_ids, original_question_ids_matched=True, cached_files_unchanged=True,
        unchanged_config_SHA=CONFIG_SHA, unchanged_evaluation_fingerprint=FINGERPRINT,
        fingerprint_override_required=False, cache_source_provenance_present=bool(cache_sources),
        probe_source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), GPU_used=False, API_requests_made=0)
    args.output.write_text(json.dumps(result, indent=2) + '\n'); print(json.dumps(result))


if __name__ == '__main__':
    main()
