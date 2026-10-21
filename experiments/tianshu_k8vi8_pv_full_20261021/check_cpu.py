# SPDX-License-Identifier: Apache-2.0
"""Current-profile identity, real CPU layout, and two independently written PV references."""
import ast,hashlib,importlib.util,inspect,json,tempfile,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from plans import PLANS,MODES,CASES,CURRENT_SHA
from layout import allocate,PARENT as LAYOUT_PARENT
from reference_numpy import Oracle,quantize
import reference_torch
from verification import compare
import source
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'tianshu_k8vi8_pv_int8_20261021'
V8=BASE.parent/'tianshu_k8vi8_heads_v2_20261021'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert not torch.cuda.is_initialized();torch.set_num_threads(4)
for path in BASE.glob('*.py'):ast.parse(path.read_text())
for name in ('source.py','kernel_pv.py'):assert sha(BASE/name)==sha(OLD/name)
for name in ('writer.py','reference_quantize.py'):assert sha(BASE/name)==sha(V8/name)
primitive=read(OLD/'PRIMITIVE_GPU_GATE_32g.json')
assert primitive['complete'] and primitive['independently_CPU_rechecked'] and primitive['all_guards_inputs_and_RNG_unchanged']
for path,digest in primitive['frozen_sources'].items():assert sha(path)==digest
text,proof=source.render(source.SOURCE.read_bytes());compile(text,'<full-PV-reader>','exec');assert sha(source.SOURCE)==CURRENT_SHA
reference=json.loads(subprocess.check_output([sys.executable,str(BASE/'check_references_cpu.py')],text=True))
assert reference['complete'] and reference['CUDA_not_initialized_before_or_after']
tests=reference['reference_vectors'];writer_values=reference['writer_fixture_elements']
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache
layouts=[]
for mode in MODES:
 for strided in (False,True):
    raw,native,k,v,ks,vs,layout=allocate(torch,FullAttentionSpec,_reshape_attention_kv_cache,mode,10,strided,device='cpu')
    expected=torch.full_like(raw,17)
    for plane,value,offset in ((k,3,0),(v,5,4096),(ks,7,8192),(vs,11,8320)):
        if plane is None:continue
        plane.fill_(value);size=plane.element_size();data=plane[0].contiguous().view(torch.uint8).flatten()
        for row in range(plane.shape[0]):expected[row*native.stride(0)+offset:row*native.stride(0)+offset+data.numel()]=data
        assert torch.equal(raw,expected)
    layouts.append(layout)
parent_path=BASE.parent/'tianshu_prefill_int8_service_20261021/candidate_profile.py'
spec=importlib.util.spec_from_file_location('_full_PV_actual_current_profile',parent_path)
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
root=Path(tempfile.mkdtemp(prefix='20261021_k8vi8_pv_full_CPU_',dir='/workspace/benchmark-results'))
command,environment,profile,sources=parent.prepare(root,'candidate')
assert len(sources)==407 and len(profile)==62 and len(command)==13 and CURRENT_SHA in sources.values()
qualified=read(BASE.parent/'tianshu_prefill_int8_official_results_20261021/OFFICIAL_GATE_32g.json')
current=read(Path(qualified['raw_root'])/'comparison.json')['profile']
assert qualified['complete'] and qualified['math_accuracy']>.95 and current['active_decode_source_sha256']==CURRENT_SHA
launches=[v for v in current['pair_load_record']['launches'].values() if v['sequences']==64]
assert launches and all(v['segments']==4 and v['tile']==16 and v['warps']==1 for v in launches)
framework_context_initialized=bool(torch.cuda.is_initialized())
paths=[*BASE.glob('*.py'),*BASE.glob('*.sh'),LAYOUT_PARENT,source.SOURCE,OLD/'source.py',OLD/'kernel_pv.py',OLD/'CPU_GATE_32g.json',OLD/'PRIMITIVE_GPU_GATE_32g.json',V8/'source.py',V8/'writer.py',V8/'reference_quantize.py',V8/'CPU_GATE_32g.json',V8/'GPU_GATE_32g.json',BASE.parent/'tianshu_soft8_mixed_planes_20261019/writer.py',Path('/usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_attention_helpers.py'),Path(inspect.getsourcefile(FullAttentionSpec)),Path(inspect.getsourcefile(_reshape_attention_kv_cache))]
paths.append(BASE/'CPU_REFERENCE_GATE_32g.json')
sources.update({str(p):sha(p) for p in paths})
gate=dict(complete=True,CPU_only=True,no_GPU_started=True,full_attention_GPU_not_run=True,model_quality_not_evaluated=True,
    original_integer_PV_renderer_and_helper_byte_identical=True,writer_and_RNE_reference_byte_identical=True,current_407_profile_verified=True,environment_count=62,
    actual_current_reader_SHA=CURRENT_SHA,actual_current_B64_launch=dict(segments=4,tile=16,warps=1),native_CPU_layout_checks=layouts,
    independent_reference_vectors=tests,CPU_writer_fixture_elements=writer_values,default_CPU_RNG_unchanged=True,CPU_reference_did_not_initialize_GPU=True,framework_import_initialized_device_context=framework_context_initialized,no_candidate_GPU_kernel_executed=True,
    declared_integer_algorithm_atol=.001,declared_integer_algorithm_rtol=.005,declared_integer_relative_L2_max=.01,
    ideal_attention_approximation_error_retained=True,final_model_quality_requires_new_Math_over95=True,source_proof=proof,sources=sources,CPU_profile_root=str(root))
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n')
print(json.dumps({k:v for k,v in gate.items() if k not in ('native_CPU_layout_checks','sources','source_proof')}))
