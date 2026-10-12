# SPDX-License-Identifier: Apache-2.0
"""GQA-packed native 2D tiles versus verified FA2 tile128, isolated process."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / 'result.json').exists()
    deadline = datetime.fromisoformat(json.loads(Path(
        '/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json'
    ).read_text())['stop_at_utc'])
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 0
    signal.alarm(min(1800, remaining))
    assert 'No running processes found' in subprocess.check_output(['ixsmi'], text=True)
    import torch
    import vllm_fl
    from flag_gems import runtime
    import vllm.v1.attention.backends.triton_attn as backend
    import vllm.v1.attention.ops.triton_unified_attention as native
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    fixture_path = Path(__file__).resolve().parents[1] / 'tianshu_native_attention/validate_hybrid.py'
    spec = importlib.util.spec_from_file_location('tianshu_native_prefill_fixture', fixture_path)
    fixture = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixture
    spec.loader.exec_module(fixture)
    original_forward = backend.TritonAttentionImpl.forward
    original_kernel = native.kernel_unified_attention
    original_lookup = runtime.get_heuristic_config
    hybrid_source = Path('/usr/local/lib/python3.12/site-packages/vllm_fl/worker/iluvatar_hybrid_attention.py')
    hybrid = fixture.load_hybrid(hybrid_source)
    os.environ.update(VLLM_FL_ILUVATAR_HYBRID_ATTENTION='1', VLLM_FL_USE_FLAGGEMS_ATTN='0', FLAGGEMS_ILUVATAR_COMPACT_VARLEN='1')
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(hidden_size=2048, num_hidden_layers=42,
        num_attention_heads=16, num_key_value_heads=2), dtype=torch.bfloat16, quantization=None, is_multimodal_model=False),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1),
        lora_config=None, speculative_config=None)
    assert hybrid.configure(config, logging.getLogger('native_prefill_probe'))

    def selected_lookup(name):
        if name != 'mha_block_128':
            return original_lookup(name)
        return {key: (lambda args, value=value: value) for key, value in
                dict(BLOCK_M=128, BLOCK_N=128, num_warps=8, num_stages=2).items()}
    runtime.get_heuristic_config = selected_lookup
    cache = {}
    original_reference = fixture.reference
    cpu_checks = []

    def cached_reference(q, k, v, table, qlens, lengths):
        key = tuple(qlens), tuple(lengths), tuple(q.stride()), tuple(k.stride())
        if key not in cache:
            cache[key] = original_reference(q, k, v, table, qlens, lengths)
            begin = 0
            for seq, (ql, kl) in enumerate(zip(qlens, lengths)):
                ids = table[seq, :(kl + 15) // 16].long()
                kc = k[ids].cpu().reshape(-1, 2, 128)[:kl].float()
                vc = v[ids].cpu().reshape(-1, 2, 128)[:kl].float()
                positions = sorted({0, min(127, ql - 1), ql - 1})
                for head in (0, 7, 8, 15):
                    qc = q[begin + torch.tensor(positions, device=q.device), head].cpu().float()
                    score = (qc @ kc[:, head // 8].T) * 128 ** -.5
                    score.masked_fill_(torch.arange(kl)[None, :] > kl - ql + torch.tensor(positions)[:, None], -float('inf'))
                    cpu = score.softmax(-1) @ vc[:, head // 8]
                    expected = cache[key][begin + torch.tensor(positions, device=q.device), head].cpu()
                    check = fixture.error(expected, cpu)
                    cpu_checks.append(check)
                    assert check['passed'], 'Independent CPU oracle failed'
                begin += ql
        return cache[key]
    fixture.reference = cached_reference
    plans = [(64, 128, 8, 2), (128, 64, 8, 2), (128, 128, 8, 2),
             (128, 128, 8, 1), (128, 128, 16, 2), (256, 128, 16, 1)]
    scenarios = [([2048], [2048]), ([2048], [4096]), ([4096], [4096]),
                 ([513, 2048], [4096, 4096]), ([2048], [16384])]
    data = {'complete': False, 'official_benchmark': False, 'adoption_allowed': False,
            'baseline': 'verified FA2 BM128/BN128/w8/s2', 'cases': [], 'cpu_checks': cpu_checks,
            'native_sha256': hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            'hybrid_sha256': hashlib.sha256(hybrid_source.read_bytes()).hexdigest()}

    def save():
        (root / 'result.json').write_text(json.dumps(data, indent=2) + '\n')

    try:
        with torch.inference_mode():
            for scenario_id, (qlens, lengths) in enumerate(scenarios):
                for bm, bn, warps, stages in plans:
                    outputs = {}
                    routes = []

                    class Router:
                        def __getitem__(self, grid):
                            def launch(*args, **kwargs):
                                assert not args and kwargs['IS_3D'] is False
                                block_q = bm // kwargs['num_queries_per_kv']
                                new_grid = (kwargs['query_ptr'].shape[0] // block_q + kwargs['num_seqs'], grid[1])
                                kwargs.update(BLOCK_M=bm, BLOCK_Q=block_q, TILE_SIZE=bn, num_warps=warps, num_stages=stages)
                                if (grid, new_grid) not in routes:
                                    routes.append((grid, new_grid))
                                return original_kernel[new_grid](**kwargs)
                            return launch
                    router = Router()

                    def candidate_native(*args, **kwargs):
                        out = args[7] if len(args) > 7 else kwargs['output']
                        metadata = args[6] if len(args) > 6 else kwargs['attn_metadata']
                        outputs[out.data_ptr()] = (out, metadata.num_actual_tokens)
                        old = native.kernel_unified_attention
                        native.kernel_unified_attention = router
                        try:
                            return original_forward(*args, **kwargs)
                        finally:
                            native.kernel_unified_attention = old
                    entry = {'query_lengths': qlens, 'kv_lengths': lengths,
                             'configuration': dict(BLOCK_M=bm, TILE_SIZE=bn, num_warps=warps, num_stages=stages)}
                    try:
                        result = fixture.run_case(hybrid, candidate_native, f'scenario{scenario_id}_m{bm}_n{bn}_w{warps}_s{stages}',
                                                  qlens, lengths, True, 2026101600 + scenario_id)
                        entry.update(result, candidate_native_vs_fp32=result['native_vs_fp32'],
                                     verified_fa2_vs_fp32=result['candidate_vs_fp32'],
                                     candidate_native_ms=result['median_ms']['original'],
                                     verified_fa2_ms=result['median_ms']['hybrid'],
                                     candidate_output_tail_preserved=all(bool((out[count:] == 17.).all()) for out, count in outputs.values()),
                                     routing_grids=routes)
                        entry['change_percent'] = 100 * (entry['candidate_native_ms'] / entry['verified_fa2_ms'] - 1)
                    except Exception as error:
                        entry['error'] = str(error)
                        if any(term in str(error).lower() for term in ('illegal memory', 'device-side assert')):
                            raise
                    data['cases'].append(entry)
                    save()
                    print(json.dumps(entry), flush=True)
                    torch.cuda.empty_cache()
        data.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat())
    finally:
        native.kernel_unified_attention = original_kernel
        runtime.get_heuristic_config = original_lookup
        save()


if __name__ == '__main__':
    main()
