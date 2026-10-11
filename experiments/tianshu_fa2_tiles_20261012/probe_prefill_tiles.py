# SPDX-License-Identifier: Apache-2.0
"""Isolated paged FA2 tile sweep using the previously verified fixture.

Only this process's heuristic lookup changes. Installed files stay untouched.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import torch
from flag_gems import runtime
import vllm.v1.attention.backends.triton_attn as backend


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hybrid-source', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'result.json').exists():
        raise RuntimeError('Refusing to overwrite an existing tile sweep')
    lease = Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json')
    deadline = datetime.fromisoformat(json.loads(lease.read_text())['stop_at_utc'])
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
    if remaining <= 0:
        raise RuntimeError('Reservation stop time reached')
    signal.alarm(min(1800, remaining))
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ['VLLM_FL_ILUVATAR_HYBRID_ATTENTION'] = '1'
    os.environ['VLLM_FL_USE_FLAGGEMS_ATTN'] = '0'
    os.environ['FLAGGEMS_ILUVATAR_COMPACT_VARLEN'] = '1'
    fixture_path = Path(__file__).parents[1] / 'tianshu_native_attention/validate_hybrid.py'
    fixture = load_module(fixture_path, 'tianshu_fa2_tile_fixture')
    hybrid = fixture.load_hybrid(args.hybrid_source)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(hidden_size=2048, num_hidden_layers=42,
                                      num_attention_heads=16, num_key_value_heads=2),
            dtype=torch.bfloat16, quantization=None, is_multimodal_model=False),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1,
                                        data_parallel_size=1, decode_context_parallel_size=1),
        lora_config=None, speculative_config=None,
    )
    if not hybrid.configure(config, logging.getLogger('fa2_tile_probe')):
        raise RuntimeError('Hybrid fixture did not configure')
    wrapped_forward = backend.TritonAttentionImpl.forward
    original_lookup = runtime.get_heuristic_config
    installed_config = original_lookup('mha_block_128')
    actual = {key: installed_config[key](()) for key in
              ('BLOCK_M', 'BLOCK_N', 'num_warps', 'num_stages')}
    assert actual == dict(BLOCK_M=128, BLOCK_N=32, num_warps=4, num_stages=3), actual
    original_reference = fixture.reference
    oracle_cache = {}
    cpu_oracle_checks = []

    def cached_reference(q, k, v, table, qlens, lengths):
        # run_case resets the fixed seed for each scenario before allocating inputs.
        key = (tuple(qlens), tuple(lengths), tuple(q.stride()), tuple(k.stride()))
        if key not in oracle_cache:
            oracle_cache[key] = original_reference(q, k, v, table, qlens, lengths)
            begin = 0
            for seq, (ql, kl) in enumerate(zip(qlens, lengths)):
                pages = table[seq, :(kl + 15) // 16].long()
                keys = k[pages].reshape(-1, 2, 128)[:kl].float().cpu()
                values = v[pages].reshape(-1, 2, 128)[:kl].float().cpu()
                positions = sorted(set((0, min(127, ql - 1), ql - 1)))
                for head in (0, 7, 15):
                    queries = q[begin + torch.tensor(positions, device=q.device), head].float().cpu()
                    scores = queries @ keys[:, head // 8].T * 128 ** -.5
                    mask = torch.arange(kl)[None, :] <= (
                        kl - ql + torch.tensor(positions)[:, None])
                    scores.masked_fill_(~mask, float('-inf'))
                    expected = scores.softmax(-1) @ values[:, head // 8]
                    selected = oracle_cache[key][begin + torch.tensor(positions, device=q.device), head].cpu()
                    check = fixture.error(selected, expected)
                    check.update(sequence=seq, head=head, positions=positions,
                                 qlens=qlens, kv_lengths=lengths)
                    cpu_oracle_checks.append(check)
                    if not check['passed']:
                        del oracle_cache[key]
                        raise RuntimeError('Independent CPU oracle failed: ' + str(check))
                begin += ql
        return oracle_cache[key]

    fixture.reference = cached_reference

    def baseline_forward(*forward_args, **forward_kwargs):
        selected_lookup = runtime.get_heuristic_config
        runtime.get_heuristic_config = original_lookup
        try:
            return wrapped_forward(*forward_args, **forward_kwargs)
        finally:
            runtime.get_heuristic_config = selected_lookup

    plan = json.loads(Path(__file__).with_name('experiment_plan.json').read_text())
    result = {
        'official_benchmark': False, 'complete': False,
        'installed_files_modified': False, 'actual_baseline_config': actual,
        'hybrid_sha256': hashlib.sha256(args.hybrid_source.read_bytes()).hexdigest(),
        'fixture_sha256': hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        'cases': [],
        'cpu_oracle_checks': cpu_oracle_checks,
        'cpu_oracle_pending': True,
        'adoption_allowed': False,
    }

    def save():
        (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    with torch.inference_mode():
        try:
            for scenario_index, scenario in enumerate(plan['cases']):
                for tile in plan['configurations']:
                    def selected_lookup(name, tile=tile):
                        if name != 'mha_block_128':
                            return original_lookup(name)
                        return {key: (lambda args, value=value: value)
                                for key, value in tile.items()}

                    runtime.get_heuristic_config = selected_lookup
                    name = f"scenario{scenario_index}_m{tile['BLOCK_M']}_n{tile['BLOCK_N']}_w{tile['num_warps']}_s{tile['num_stages']}"
                    try:
                        case = fixture.run_case(
                            hybrid, baseline_forward, name, scenario['query_lengths'],
                            scenario['kv_lengths'], True, 2026101240 + scenario_index)
                        case['configuration'] = tile
                        case['baseline_name'] = 'default_fa2'
                        case['candidate_name'] = 'selected_fa2'
                        result['cases'].append(case)
                        print(json.dumps(case), flush=True)
                    except Exception as error:
                        failure = {'name': name, 'configuration': tile,
                                   'passed': False, 'error': str(error)}
                        result['cases'].append(failure)
                        print(json.dumps(failure), flush=True)
                        save()
                        if any(text in str(error).lower() for text in
                               ('illegal memory', 'device-side assert')):
                            raise
                    save()
            result['complete'] = True
            result['cpu_oracle_pending'] = False
            result['numerical_failures'] = sum(
                not case.get('candidate_vs_fp32', {}).get('passed', False)
                or not case.get('native_vs_fp32', {}).get('passed', False)
                for case in result['cases'])
        finally:
            runtime.get_heuristic_config = original_lookup
            save()


if __name__ == '__main__':
    main()
