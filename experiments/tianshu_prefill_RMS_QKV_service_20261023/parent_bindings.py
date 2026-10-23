# SPDX-License-Identifier: Apache-2.0
"""Read genuine original-op closure state; never replace original model sources."""
import hashlib,inspect,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;EXP=BASE.parent
PARENT=EXP/'tianshu_native_int8_attention_service_20261022/policy.py'
TORCH_REGISTRY=Path('/usr/local/lib/python3.12/site-packages/torch/_library/custom_ops.py')
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def bind_original_attention():
 from torch._library.custom_ops import OPDEFS
 proof=json.loads((BASE/'PARENT_BINDING_CPU_32g.json').read_text());assert proof['complete'] and sha(PARENT)==proof['parent_policy_sha256'] and sha(TORCH_REGISTRY)==proof['torch_registry_sha256']
 operation=OPDEFS['tianshu_native_attention::linear'];fn=operation._init_fn;assert Path(fn.__code__.co_filename).resolve()==PARENT and fn.__code__.co_name=='linear' and list(fn.__code__.co_freevars)==proof['linear_closure_variables'];closure=inspect.getclosurevars(fn).nonlocals;assert {'banks','record','save'}<=closure.keys();record=closure['record'];banks=closure['banks'];save=closure['save'];assert record['installed'] and record['mode']=='candidate' and record['source']==str(PARENT) and record['source_sha256']==sha(PARENT) and isinstance(banks,dict) and callable(save) and Path(save.__code__.co_filename).resolve()==PARENT
 return dict(operation=operation,record=record,banks=banks,save=save,original_function=fn,scope='original actual weight banks and record; any new route must identify actual MM/epilogue and must not claim original standalone A8 calls')
