# SPDX-License-Identifier: Apache-2.0
"""Exhaustively check logical-to-physical CTA bijections using archived .cv source."""
import argparse
import ast
import hashlib
import itertools
import json
from pathlib import Path

from grid_permute import CV_SHA, ORIGINAL, assignments, permute_grid, render_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.cv_source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CV_SHA
    variants = []
    configurations = 0
    coordinates = 0
    for axes in itertools.permutations(range(3)):
        rendered = render_source(raw, axes)
        ast.parse(rendered)
        assert rendered.replace(assignments(axes), ORIGINAL).encode() == raw
        for batch in (1, 2, 7, 8, 16, 24, 32, 64):
            for padded in (False, True):
                queries = sum(int(not padded or i % 3 != 1) for i in range(batch))
                segments = 8 if batch == 64 else 64 if batch == 1 else 4 if batch >= 16 else 16
                logical = (queries // 2 + batch, 2, segments)
                physical = permute_grid(logical, axes)
                expected = set(itertools.product(*(range(size) for size in logical)))
                mapped = [tuple(pid[axes[i]] for i in range(3))
                          for pid in itertools.product(*(range(size) for size in physical))]
                assert len(mapped) == len(expected) and set(mapped) == expected
                configurations += 1
                coordinates += len(mapped)
        variants.append(dict(logical_to_physical_axes=list(axes),
                             generated_sha256=hashlib.sha256(rendered.encode()).hexdigest()))
    output = dict(complete=True, cpu_only=True, gpu_execution=False,
                  baseline_cv_sha256=CV_SHA, configurations=configurations,
                  coordinates=coordinates, all_logical_ctas_preserved_exactly_once=True,
                  source_changes_exactly_three_program_id_assignments=True,
                  arithmetic_reducer_and_two_dimensional_mapping_unchanged=True,
                  variants=variants, sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (Path(__file__), Path(__file__).with_name("grid_permute.py"))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
