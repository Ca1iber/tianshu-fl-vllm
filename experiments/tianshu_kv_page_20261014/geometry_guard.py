# SPDX-License-Identifier: Apache-2.0
"""Generate only the exact guarded KV page16/page32 geometry predicate."""
import ast
import hashlib

HYBRID_SHA = '5c55a8d57e438b36d176555fd9273ab9ba665dada4c57d57c19633394c2ed4e4'


def generate(raw):
    assert hashlib.sha256(raw).hexdigest() == HYBRID_SHA
    source = raw.decode()
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'supported']
    assert len(nodes) == 1
    original = ast.get_source_segment(source, nodes[0])
    old = 'k.shape[1:] == (16,2,128)'
    assert original.count(old) == 1
    candidate = original.replace(old, 'k.shape[1:] in ((16,2,128),(32,2,128))') + '\n'
    ast.parse(candidate)
    return original + '\n', candidate


def install(module, raw):
    original, candidate = generate(raw)
    namespace = dict(module.__dict__)
    exec(compile(candidate, '<tianshu-page32-supported>', 'exec'), namespace)
    return namespace['supported'], dict(original_hybrid_sha256=HYBRID_SHA,
        original_predicate_sha256=hashlib.sha256(original.encode()).hexdigest(),
        expanded_predicate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
        supported_page_sizes=[16,32], all_other_predicates_unchanged=True), original, candidate
