# SPDX-License-Identifier: Apache-2.0
"""Exercise real custom-op fake dispatch and FX mutation, then both complete profiles."""
import argparse,ast,hashlib,importlib.util,json,sys,types
from pathlib import Path
BASE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();root=args.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir()
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
assert not torch.cuda.is_initialized();sys.path.insert(0,str(BASE))
import native_prefill_ops
from native_prefill_contract import use_native
oldlib=torch.library.Library('tianshu_m_range_prefill','DEF');oldlib.define('fused_or_original(Tensor x, Tensor weight, Tensor packed, Tensor scales, int bank_id) -> Tensor')
checks=0
for m in (1,8,64,128,512,4096):
    with FakeTensorMode():
        x=torch.empty((m,2048),dtype=torch.bfloat16);w=torch.empty((12288,2048),dtype=torch.bfloat16);packed=torch.empty((12288,2048),dtype=torch.int8);scales=torch.empty((12288,),dtype=torch.float32)
        out=torch.ops.tianshu_native_prefill.fused_or_original.default(x,w,packed,scales,7)
        assert out.shape==(m,6144) and out.dtype==x.dtype and out.device==x.device and out is not x
    checks+=1
symbol=ShapeEnv().create_unbacked_symint();out=native_prefill_ops.fake(torch.empty((symbol,2048),device='meta',dtype=torch.bfloat16),None,None,None,7);assert out.shape[0] is symbol and out.shape[1]==6144;checks+=1
try:native_prefill_ops.fake(torch.empty((3,1024),device='meta',dtype=torch.bfloat16),None,None,None,7)
except AssertionError:checks+=1
else:raise AssertionError('wrong input width accepted by fake')
class StubPass:
    def hash_dict(self,value):return str(value)
class Parent:
    def __init__(self):self.calls=0
    def __call__(self,graph):self.calls+=1
    def uuid(self):return 'inherited-parent-runtime-separately-audited'
context=types.SimpleNamespace(compile_range=types.SimpleNamespace(start=1,end=4096));text=(BASE/'native_prefill_fx.py').read_text();needle='from vllm.compilation.passes.inductor_pass import InductorPass,get_pass_context\n';assert text.count(needle)==1
namespace=dict(__file__=str(BASE/'native_prefill_fx.py'),InductorPass=StubPass,get_pass_context=lambda:context);exec(compile(text.replace(needle,''),str(BASE/'native_prefill_fx.py'),'exec'),namespace);NativePass=namespace['NativePrefillPass'];fxchecks=0
for mode in ('control','candidate'):
    for bounds in ((1,4096),(64,64),(1,128)):
        for bank in (0,41):
            graph=torch.fx.Graph();x,w,packed,s=[graph.placeholder(name) for name in ('x','weight','packed','scales')];node=graph.call_function(torch.ops.tianshu_m_range_prefill.fused_or_original.default,(x,w,packed,s,bank));node.meta=dict(val=torch.empty((symbol,6144),device='meta',dtype=torch.bfloat16),sentinel='preserve-meta');clone=graph.call_function(torch.ops.aten.clone.default,(node,));graph.output(clone)
            before=graph.python_code('self').src;metadata=node.meta;parent=Parent();context.compile_range.start,context.compile_range.end=bounds;NativePass(parent,mode,root/f'fx_{mode}_{bounds[0]}_{bounds[1]}_{bank}')(graph);graph.lint();assert parent.calls==1 and node.meta is metadata and node.args[-1]==bank and clone.target==torch.ops.aten.clone.default
            assert node.target==(torch.ops.tianshu_native_prefill.fused_or_original.default if bounds==(1,4096) else torch.ops.tianshu_m_range_prefill.fused_or_original.default)
            if bounds!=(1,4096):assert before==graph.python_code('self').src
            fxchecks+=1
for m in (1,8,64,128,511,512,513,4095,4096):
    assert not use_native('control',m,2048,True) and not use_native('candidate',m,2048,False)
    assert not use_native('candidate',m,1024,True)
assert use_native('candidate',512,2048,True) and use_native('candidate',4096,2048,True)
spec=importlib.util.spec_from_file_location('_native_CPU_profile',BASE/'candidate_profile.py');profile=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile);arms={}
for mode in ('control','candidate'):
    armroot=root/mode;armroot.mkdir();command,env,selected,sources=profile.prepare(armroot,mode,check_gate=False);assert len(command)==13 and len(selected)==62 and len({**selected,'TRITON_CACHE_DIR':env['TRITON_CACHE_DIR']})==63
    arms[mode]=dict(command=command,environment=selected,sources=sources,cache_environment={'TRITON_CACHE_DIR':env['TRITON_CACHE_DIR']})
assert arms['control']['sources']==arms['candidate']['sources'] and arms['control']['command']==arms['candidate']['command'] and arms['control']['cache_environment']==arms['candidate']['cache_environment']
normalized={mode:{key:value.replace(str(root/mode),'__ARM_ROOT__') for key,value in arm['environment'].items()} for mode,arm in arms.items()};differences={key for key in normalized['control'] if normalized['control'][key]!=normalized['candidate'][key]};assert differences=={'PYTHONPATH'}
for path in BASE.rglob('*.py'):ast.parse(path.read_text(),str(path))
assert not torch.cuda.is_initialized() and 'flag_gems' not in sys.modules
report=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,real_fake_dispatch_and_symbolic_cases=checks,fake_device_scope='CPU FakeTensor and symbolic Meta; real CUDA compiler still needs smoke',FX_context_stubbed_only_for_CPU_logic=True,actual_FX_rewrite_cases=fxchecks,parent_and_metadata_preserved=True,decode_and_unknown_M_delegation_contract_checked=True,both_complete_profiles_prepared=True,inherited_environment_keys=62,additional_explicit_cache_keys=1,total_selected_environment_keys=63,original_server_argv_count=13,numerical_source_maps_identical=True,only_normalized_environment_difference='PYTHONPATH',CPU_gate_excluded_from_own_prepare_to_avoid_hash_cycle=True,source_count_before_CPU_gate=len(arms['candidate']['sources']),raw_directory=str(root),sources=arms['candidate']['sources'])
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(report,indent=2)+'\n');(root/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({key:value for key,value in report.items() if key!='sources'}))
