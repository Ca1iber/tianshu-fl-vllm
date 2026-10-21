# SPDX-License-Identifier: Apache-2.0
"""Exercise native prefix/search arithmetic exhaustively for ragged Q1/Q2."""
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
import source

base = Path(__file__).resolve().parent
assert 'torch' not in sys.modules
text, proof = source.generate()
patterns = [list(p) for n in range(1, 9) for p in itertools.product((1, 2), repeat=n)]
rng = random.Random(20261021)
patterns += [[rng.choice((1, 2)) for _ in range(n)] for n in (16, 32, 64) for _ in range(128)]
covered = 0
for lengths in patterns:
    cu = [0]
    for value in lengths:
        cu.append(cu[-1] + value)
    boundaries = [x // 2 + i for i, x in enumerate(cu)]
    seen = []
    for block in range(cu[-1] // 2 + len(lengths)):
        seq = max(i for i, boundary in enumerate(boundaries) if boundary <= block)
        assert seq < len(lengths)
        local = block - boundaries[seq]
        if local * 2 >= lengths[seq]:
            continue
        seen += list(range(cu[seq] + local * 2, min(cu[seq] + (local + 1) * 2, cu[seq + 1])))
    assert sorted(seen) == list(range(cu[-1])) and len(set(seen)) == cu[-1], lengths
    for token in range(cu[-1]):
        seq = max(i for i, boundary in enumerate(cu) if boundary <= token)
        assert seq < len(lengths) and cu[seq] <= token < cu[seq+1]
    covered += cu[-1]
paths = [base/'source.py', base/'launch.py', base/'check_cpu.py', base/'probe_gpu.py',
    base/'run_component.sh', base/'verify_cpu.py', base/'PLAN_32g.md', source.PARENT,
    base.parent/'tianshu_k8v16_pipeline_20261019/bounded_prefill.py',
    base.parent/'tianshu_k8v16_pipeline_20261019/dense_prefill.py',
    base.parent/'tianshu_k8v16_pipeline_20261019/value_planes.py',
    Path('/usr/local/lib/python3.12/site-packages/vllm_fl/worker/iluvatar_hybrid_attention.py'),
    Path('/usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_attention_helpers.py')]
result = dict(complete=True, CPU_only=True, proof=proof, patterns=len(patterns), query_tokens_checked=covered,
    native_block_Q2_prefix_and_reduce_token_mapping_checked=True,
    GPU_correctness_proven=False, model_or_service_executed=False, adoption_allowed=False,
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
(base/'CPU_GATE_32g.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('sources',)}, indent=2))
