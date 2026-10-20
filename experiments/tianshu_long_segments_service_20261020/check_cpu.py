# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,importlib.util,json
from pathlib import Path
from types import SimpleNamespace
from policy import select,bind
BASE=Path(__file__).resolve().parent
p=BASE.parent/'tianshu_k8v16_pipeline_20261019/dispatch_policy.py'
s=importlib.util.spec_from_file_location('qualified_policy',p);original=importlib.util.module_from_spec(s);s.loader.exec_module(original)
selected=set(range(33,57));candidate=select(original.choose_config,selected)
checks=0
for batch in range(1,70):
    for length in (1,127,128,129,4096,8192,8193,16384,131072):
        before=original.choose_config(batch,length);after=candidate(batch,length)
        expected=None if before is None else dict(before,segments=16) if batch in selected else before
        assert after==expected
        assert original.choose_config(batch,length)==before
        checks+=1
old_a,old_b,new_a,new_b=object(),object(),object(),object()
producer=SimpleNamespace(kernel=old_a)
reducer=SimpleNamespace(kernel=old_b,producer=producer)
reader=SimpleNamespace(kernel_unified_attention=producer,reduce_segments=reducer,unified_attention=object())
unified=reader.unified_attention
legacy=SimpleNamespace(choose_config=original.choose_config)
bind(reader,producer,SimpleNamespace(kernel_unified_attention=new_a,reduce_segments=new_b),legacy,selected)
assert producer.kernel is new_a and reducer.kernel is new_b and reader.unified_attention is unified
assert legacy.choose_config(40,131072)['segments']==16
for path in BASE.rglob('*.py'):ast.parse(path.read_text())
policy=json.loads((BASE/'batch_policy.json').read_text());raw=Path(policy['source_raw'])
assert hashlib.sha256(raw.read_bytes()).hexdigest()==policy['source_sha256']
x=json.loads(raw.read_text());assert x['complete'] and len(x['cases'])==44
result=dict(complete=True,CPU_only=True,GPU_service_executed=False,adopted=False,selection_cases=checks,
            original_configuration_not_mutated=True,only_two_kernel_objects_and_choose_function_rebound=True,
            original_python_dispatch_object_preserved=True,selected_batches=list(range(33,57)),
            generated_source_expected_sha256='7b94b63e0b33a2775b9c0cd043b51120fbbb6b35a81f5134897e00fbea4cf106',
            sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.rglob('*.py')})
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='sources'}))
