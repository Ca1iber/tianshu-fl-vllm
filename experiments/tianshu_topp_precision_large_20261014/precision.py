# SPDX-License-Identifier: Apache-2.0
"""Owned standalone TopP repair; preserve native TopK branches and source provenance."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

SOURCE_SHA = "64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8"


def topk_ast(text):
    tree = ast.parse(text)
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_topk_topp_kernel")
    row = next(n for n in kernel.body if isinstance(n, ast.For))
    branch = next(n for n in row.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "TOPK_ENABLED")
    return ast.dump(branch, include_attributes=False)


def render_source(raw):
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
    original = raw.decode()
    text = original
    edits = []

    def edit(before, after, count=1):
        nonlocal text
        assert text.count(before) == count
        text = text.replace(before, after)
        edits.append((before, after, count))

    begin = text.index("@triton.jit\ndef _update_min_larger_stats(")
    end = text.index("@triton.jit\ndef _topk_topp_kernel(", begin)
    helper = text[begin:end].replace("def _update_min_larger_stats(", "def _update_min_larger_stats_exact(")
    helper = helper.replace("tl.abs(data - tile_min) < 1e-9", "data == tile_min")
    helper = helper.replace("tl.abs(tile_min - min_larger) < 1e-9", "tile_min == min_larger")
    edit("@triton.jit\ndef _topk_topp_kernel(", helper + "@triton.jit\ndef _topk_topp_kernel(")
    edit('\n        final_pivot = -float("inf")\n',
         '\n        top_p_prob_shift = 0.0\n        top_p_prob_sum = 1.0\n'
         '        top_p_prob_pivot = 1.0\n        top_p_prob_min = 1.0\n'
         '        final_pivot = -float("inf")\n')
    start = text.index('        if TOPP_ENABLED and final_pivot == -float("inf"):')
    finish = text.index("        # Sixth pass:", start)
    standalone = text[start:finish]
    repaired = standalone.replace("_update_min_larger_stats(", "_update_min_larger_stats_exact(")
    assert standalone.count("_update_min_larger_stats(") == 2
    stop = "(max_range - min_range) < 1e-9 or num_iters >= 18"
    assert standalone.count(stop) == 2
    repaired = repaired.replace(stop, "(max_range - min_range) < 1e-12 or num_iters >= 32")
    needle = "                min_range = tl.exp(min_logit - max_sample) / sum_exp_logits\n"
    assert repaired.count(needle) == 1
    repaired = repaired.replace(needle, needle + "                min_range = tl.where(min_range >= max_range, 0.0, min_range)\n")
    pivot = "                final_pivot = tl.log(p_pivot * sum_exp_logits) + max_sample\n"
    assert repaired.count(pivot) == 1
    repaired = repaired.replace(pivot, pivot +
        "                top_p_prob_shift = max_sample\n                top_p_prob_sum = sum_exp_logits\n"
        "                top_p_prob_pivot = p_pivot\n                top_p_prob_min = min_larger_prob\n")
    edit(standalone, repaired)
    edit("                keep_mask = (logits_blk > final_pivot) & mask_n\n",
         "                if TOPP_ENABLED and not TOPK_ENABLED:\n"
         "                    filter_prob = tl.exp(logits_blk - top_p_prob_shift) / top_p_prob_sum\n"
         "                    keep_mask = (filter_prob > top_p_prob_pivot) & mask_n\n"
         "                else:\n                    keep_mask = (logits_blk > final_pivot) & mask_n\n")
    edit("                    duplicate_mask = (\n                        tl.abs(logits_blk - duplicate_logit) < 1e-9\n                    ) & mask_n\n",
         "                    if TOPP_ENABLED and not TOPK_ENABLED:\n"
         "                        duplicate_mask = (filter_prob == top_p_prob_min) & mask_n\n"
         "                    else:\n                        duplicate_mask = (\n"
         "                            tl.abs(logits_blk - duplicate_logit) < 1e-9\n                        ) & mask_n\n")
    assert topk_ast(text) == topk_ast(original)
    restored = text
    for before, after, count in reversed(edits):
        assert restored.count(after) == count
        restored = restored.replace(after, before)
    assert restored.encode() == raw
    return text


def load_kernel(native_source, directory):
    raw = Path(native_source).read_bytes()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "generated_topp_precision.py"
    source.write_text(render_source(raw))
    name = "_tianshu_topp_precision"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module._topk_topp_kernel, dict(native_source_sha256=SOURCE_SHA,
        generated_source=str(source), generated_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        native_TopK_branch_AST_unchanged=True, reversible_to_exact_native_source=True,
        installed_source_modified=False, standalone_duplicate_probability_equality=True,
        standalone_search_limit=32, standalone_probability_interval_tolerance=1e-12,
        standalone_final_mask_in_probability_domain=True)
