# SPDX-License-Identifier: Apache-2.0
import ast, hashlib, json
from pathlib import Path

BASE = Path(__file__).resolve().parent
SITE = Path('/usr/local/lib/python3.12/site-packages')
OLD = BASE.parent/'tianshu_dspark_20261012'
DEPENDENCIES = [OLD/'tianshu_dspark_adapter/proposer.py',
    SITE/'vllm/v1/spec_decode/dflash.py', SITE/'vllm/v1/spec_decode/llm_base_proposer.py',
    SITE/'vllm/v1/spec_decode/utils.py', SITE/'vllm/v1/attention/backend.py',
    SITE/'vllm_fl/worker/model_runner.py', SITE/'vllm/model_executor/models/llama.py',
    SITE/'vllm/model_executor/models/interfaces.py',
    Path('/workspace/MiniCPM5-2B-DSpark/config.json')]

CASES = [
 dict(name='ctx1',context=[1],past=[0],rejected=None),
 dict(name='page_cross',context=[3],past=[14],rejected=None),
 dict(name='ctx17_rejected',context=[17],past=[31],rejected=[2]),
 dict(name='ctx257_multiblock',context=[257],past=[8191],rejected=[5]),
 dict(name='ctx513_multiblock',context=[513],past=[16380],rejected=None),
 dict(name='mixed_b8',context=[1,3,17,32,63,128,256,257],past=[15,29,31,48,61,128,511,1023],rejected=[0,2,2,1,3,2,0,5]),
 dict(name='decode_b64',context=[3]*64,past=[4093+i%17 for i in range(64)],rejected=[i%3 for i in range(64)]),
 dict(name='near_limit',context=[3],past=[131061],rejected=None),
 dict(name='padded_near_limit',context=[7],past=[131060],rejected=[3]),
 dict(name='ctx256_grid_b2',context=[256,256],past=[15,2047],rejected=[1,0]),
]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def actual_method(path, cls, name):
    tree = ast.parse(Path(path).read_text())
    owner = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == cls)
    node = next(x for x in owner.body if isinstance(x, ast.FunctionDef) and x.name == name)
    assert not node.decorator_list
    module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    ns = {}; exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ns)
    return ns[name]

def manifest():
    files = sorted([*BASE.glob('*.py'), *BASE.glob('*.sh'), BASE/'PLAN.md', *DEPENDENCIES])
    return {str(p):sha(p) for p in files}

def oracle(context, past, rejected, bonus, pages, block_size=16):
    rows = len(context); rejected = rejected if rejected is not None else [0]*rows
    result = dict(input_ids=[],context_positions=[],query_positions=[],context_slots=[],query_slots=[],query_start_loc=[7*i for i in range(rows+1)],seq_lens=[],sample_indices=list(range(rows*7)))
    for i,(ctx,offset,rej) in enumerate(zip(context,past,rejected,strict=True)):
        assert 0 <= rej < ctx
        positions = list(range(offset,offset+ctx)); anchor = offset+ctx-rej
        query = list(range(anchor,anchor+7)); assert 0 <= min(query) <= max(query) < 131072
        slot = lambda p: int(pages[i][p//block_size])*block_size+p%block_size
        result['input_ids'].extend([bonus[i]]+[75982]*6)
        result['context_positions'].extend(positions); result['query_positions'].extend(query)
        result['context_slots'].extend(map(slot,positions)); result['query_slots'].extend(map(slot,query))
        result['seq_lens'].append(anchor+7)
    return result
