# SPDX-License-Identifier: Apache-2.0
import argparse,ast,base64,hashlib,importlib.util,json,os,signal,subprocess,sys,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output.resolve();assert out.is_relative_to('/workspace/benchmark-results') and not out.exists();out.mkdir();signal.alarm(300)
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
paths=[ROOT/f'tianshu_nn_silu_large_tiles{suffix}_20261021/kernel.py' for suffix in ('','_v2')]
def node(path):return next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='_fused')
assert ast.dump(node(paths[0]))==ast.dump(node(paths[1]))
selected=ROOT/'tianshu_m_range_prefill_service_v2_20261021/selected_plan.json';selection=json.loads(selected.read_text());gatepath=Path(selection['component_gate']);gate=json.loads(gatepath.read_text());sourcefiles=[Path(__file__),BASE/'PLAN_32g.md',selected,gatepath,*paths];sources={str(path):sha(path) for path in sourcefiles}
state=dict(complete=False,runner_pid=os.getpid(),sources=sources,device_AST_equal=True,cases=[],selected={},benchmark=False,model_executed=False)
def save():
 temp=out/'result.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(out/'result.json')
save()
try:
 import torch,vllm_fl,triton
 assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
 for version,path in zip(('v1','v2'),paths):
  cachedir=out/version/'cache';cachedir.mkdir(parents=True);os.environ['TRITON_CACHE_DIR']=str(cachedir);spec=importlib.util.spec_from_file_location('_binary_origin_'+version,path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  for m in (128,512,4096):
   cfg=selection['configuration_by_M'][str(m)];q=torch.zeros((m,2048),dtype=torch.int8,device='cuda');w=torch.zeros((2048,12288),dtype=torch.int8,device='cuda');asc=torch.ones(m,device='cuda');ws=torch.ones(12288,device='cuda');result=torch.full((m,6144),float('nan'),dtype=torch.bfloat16,device='cuda');integer=torch.empty((m,12288),dtype=torch.int32,device='cuda');projected=torch.empty((m,12288),dtype=torch.bfloat16,device='cuda');cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
   compiled=module._fused[(triton.cdiv(m,cfg['bm'])*triton.cdiv(6144,cfg['bn']),)](q,w,asc,ws,result,integer,projected,m,6144,2048,cfg['bm'],cfg['bn'],cfg['bk'],False,False,cfg['group_m'],num_warps=cfg['warps'],num_stages=1);torch.cuda.synchronize();assert bool((result==0).all()) and torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),gpu_rng)
   actual=hashlib.sha256(compiled.kernel).hexdigest();expected=gate['actual_timed_candidate_kernels'][str(m)];folder=cachedir/base64.b32encode(bytes.fromhex(compiled.hash)).decode().rstrip('=');record=dict(M=m,version=version,cache_hash=compiled.hash,binary_sha256=actual,expected_binary_sha256=expected['binary_sha256'],expected_full_binary_matched=actual==expected['binary_sha256'],zero_output_RNG_preserved=True,compiled_cache_directory=str(folder),use_sme=compiled.metadata.use_sme)
   state['cases'].append(record)
   if record['expected_full_binary_matched']:state['selected'][str(m)]=record
   save();print(json.dumps(record),flush=True);del q,w,asc,ws,result,integer,projected;torch.cuda.empty_cache()
 assert set(state['selected'])=={'128','512','4096'} and all(sha(path)==digest for path,digest in sources.items());state['complete']=True;save()
except BaseException:state['error']=traceback.format_exc();save();raise
finally:(BASE/'GPU_BINARY_GATE_32g.json').write_text(json.dumps(state,indent=2)+'\n')
