# SPDX-License-Identifier: Apache-2.0
"""Change only the pivot proposal; retain exact stopping and native TopK."""
import hashlib
import importlib.util
from pathlib import Path
import sys

SOURCE_SHA = '64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8'
GEO_SHA = '09156e66b6aa2f4460423ea5b8d70bee043022bd75f0a784417455d0e979479d'
parent = Path(__file__).resolve().parents[1]/'tianshu_topp_geometric_20261016/precision.py'
spec = importlib.util.spec_from_file_location('_topp_interpolation_geometric_base', parent)
previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
topk_ast = previous.topk_ast


def render_source(raw, mode):
    assert mode in ('geo', 'secant', 'illinois')
    source = previous.render_source(raw, 'geo')
    assert hashlib.sha256(source.encode()).hexdigest() == GEO_SHA
    if mode == 'geo':
        return source
    start = source.index('        if TOPP_ENABLED and final_pivot == -float("inf"):')
    finish = source.index('        # Sixth pass:', start)
    section = source[start:finish]
    edits = []

    def edit(before, after, count):
        nonlocal section
        assert section.count(before) == count
        section = section.replace(before, after)
        edits.append((before, after, count))

    initial = '                    found_pivot = 0\n'
    assert section.count(initial) == 2
    initial_outlier = ('                    interpolation_lo = sum_outlier_probs - p\n'
        '                    interpolation_hi = -p\n'
        '                    interpolation_side = 0\n')
    initial_full = ('                    interpolation_lo = 1.0 - p\n'
        '                    interpolation_hi = -p\n'
        '                    interpolation_side = 0\n')
    section = section.replace(initial, initial_outlier+initial, 1)
    second = section.rindex(initial)
    section = section[:second]+initial_full+section[second:]
    midpoint = '                        p_pivot_0 = tl.where(min_range > 0.0, geometric_middle, arithmetic_middle)\n'
    proposal = midpoint + '''                        fraction = interpolation_lo / (interpolation_lo - interpolation_hi)
                        fraction = tl.minimum(0.9, tl.maximum(0.1, fraction))
                        log_lo = tl.log(tl.maximum(min_range, 1.0e-30))
                        log_hi = tl.log(tl.maximum(max_range, 1.0e-30))
                        interpolated = tl.exp(log_lo + fraction * (log_hi - log_lo))
                        valid_interpolation = ((min_range > 0.0) & (interpolation_lo > 0.0)
                            & (interpolation_hi < 0.0) & (interpolated > min_range)
                            & (interpolated < max_range))
                        p_pivot_0 = tl.where(valid_interpolation, interpolated, p_pivot_0)
'''
    edit(midpoint, proposal, 2)
    update = '''                        if p_pivots_sum_0 > p:
                            min_range = p_pivot_0
                        elif p_pivots_sum_0 < p:
                            max_range = p_pivot_0
'''
    low_adjust = ('                            if interpolation_side == 1:\n'
                  '                                interpolation_hi *= 0.5\n') if mode == 'illinois' else ''
    high_adjust = ('                            if interpolation_side == -1:\n'
                   '                                interpolation_lo *= 0.5\n') if mode == 'illinois' else ''
    replacement = ('                        if p_pivots_sum_0 > p:\n'
                   '                            min_range = p_pivot_0\n'
                   '                            interpolation_lo = p_pivots_sum_0 - p\n' + low_adjust +
                   '                            interpolation_side = 1\n'
                   '                        elif p_pivots_sum_0 < p:\n'
                   '                            max_range = p_pivot_0\n'
                   '                            interpolation_hi = p_pivots_sum_0 - p\n' + high_adjust +
                   '                            interpolation_side = -1\n')
    edit(update, replacement, 2)
    restored = section
    for before, after, count in reversed(edits):
        assert restored.count(after) == count
        restored = restored.replace(after, before)
    for insertion in (initial_outlier, initial_full):
        assert restored.count(insertion) == 1
        restored = restored.replace(insertion, '')
    assert restored == source[start:finish]
    generated = source[:start]+section+source[finish:]
    assert topk_ast(generated) == topk_ast(raw.decode())
    assert generated.count('if (max_range - min_range) < 1e-12 or num_iters >= 32:') == 2
    compile(generated, '<interpolated_top_p>', 'exec')
    return generated


def load_kernel(native_source, directory):
    raw = Path(native_source).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    kernels, records = {}, {}
    for mode in ('geo', 'secant', 'illinois'):
        path = directory/f'generated_{mode}.py'
        path.write_text(render_source(raw, mode))
        name = '_tianshu_interpolated_topp_v2_'+mode
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
        kernels[mode] = module._topk_topp_kernel
        records[mode] = dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return kernels, dict(native_source_sha256=SOURCE_SHA, base_geo_sha256=GEO_SHA,
        modes=records, strict_CPU_tolerance_unchanged=1e-6, p_unchanged=True,
        RNG_implementation_unchanged=True, all_GPU_arithmetic_FP32=True,
        standalone_search_limit_unchanged=32, standalone_interval_tolerance_unchanged=1e-12,
        native_TopK_AST_unchanged=True, installed_source_modified=False)
