# SPDX-License-Identifier: Apache-2.0
"""Actual encoded writer/forward with a controlled soft8 prefill tile scan."""
import argparse,ast,hashlib,inspect,itertools,json,logging,os,runpy,signal,statistics,subprocess,sys,textwrap
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
from plans import PLANS
import source as codec_source
import routing


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True);a=p.parse_args()
 root=a.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir(parents=True)
 remaining=int((datetime.fromisoformat(a.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>900;signal.alarm(min(1800,remaining))
 before=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in before;(root/'hardware_before.txt').write_text(before)
 base=Path(__file__).resolve().parent
 prerequisite=json.loads((base/'PREREQUISITE_32g.json').read_text());assert prerequisite['complete']
 previous=Path(prerequisite['remote_result']);assert hashlib.sha256(previous.read_bytes()).hexdigest()==prerequisite['remote_result_sha256']
 previous_result=json.loads(previous.read_text());assert previous_result['complete'] and len(previous_result['cases'])==16
 for name,h in prerequisite['frozen_sources'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
 sys.path.insert(0,str(base.parent/'tianshu_gemm_topp_geo_20261016'))
 from validation_profile_new import prepare
 command,env,profile,frozen=prepare(root,Path('/workspace/benchmark-results/20261016_gemm_topp_geo_service_tianshu'))
 assert len(profile)==38;os.environ.clear();os.environ.update(env);runpy.run_path(str(Path(profile['PYTHONPATH'])/'sitecustomize.py'))
 import torch,vllm_fl
 import vllm.v1.attention.backends.triton_attn as backend
 import vllm.v1.attention.ops.triton_unified_attention as native
 from vllm.v1.attention.backend import AttentionType
 from vllm.v1.kv_cache_interface import KVQuantMode
 from vllm_fl.worker import iluvatar_adaptive_decode as adaptive,iluvatar_hybrid_attention as hybrid
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
 config=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(hidden_size=2048,num_hidden_layers=42,num_attention_heads=16,num_key_value_heads=2),dtype=torch.bfloat16,quantization=None,is_multimodal_model=False),parallel_config=SimpleNamespace(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1,decode_context_parallel_size=1),lora_config=None,speculative_config=None)
 assert adaptive.configure(config,logging.getLogger('soft8_prefill_tiles')) and hybrid.configure(config,logging.getLogger('soft8_prefill_tiles'))
 writer_file=Path(inspect.getsourcefile(backend.triton_reshape_and_cache_flash));cv_file=Path(sys.modules['tianshu_native_cache_policy'].__file__)
 reader,writer,proof=codec_source.load_modules(cv_file,writer_file,root/'codec_generated');route=routing.install(backend,hybrid,adaptive,reader,writer)
 fixture=Path(prerequisite['previous_probe_source']);text=fixture.read_text();tree=ast.parse(text)
 ns=dict(torch=torch,backend=backend,AttentionType=AttentionType,KVQuantMode=KVQuantMode)
 for name in ('error','reference','cpu_reference','make_impl','updates','poison','quantize','dequantize'):
  node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
  body=textwrap.dedent('\n'.join(text.splitlines()[node.lineno-1:node.end_lineno]));exec(compile(body,str(fixture),'exec'),ns)
 error,reference,cpu_reference,make_impl,updates,poison,quantize,dequantize=(ns[name] for name in ('error','reference','cpu_reference','make_impl','updates','poison','quantize','dequantize'))
 gate=json.loads((base/'CPU_GATE_32g.json').read_text());assert gate['complete'] and gate['actual_prefill_launch_configurations']==15
 for name,h in gate['sources'].items():assert hashlib.sha256((base/name).read_bytes()).hexdigest()==h
 frozen.update(prerequisite['frozen_sources'])
 for path in [base/n for n in ('source.py','routing.py','plans.py','check_cpu.py','probe.py','run_probe.py','CPU_GATE_32g.json','PREREQUISITE_32g.json')]+[fixture,Path(reader.__file__),Path(writer.__file__),cv_file,writer_file]:
  frozen[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 for name,h in frozen.items():
  target=root/'source_snapshot'/Path(name).relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes());assert hashlib.sha256(target.read_bytes()).hexdigest()==h
 d=dict(phase='preparing',complete=False,runner_pid=os.getpid(),source_snapshots=frozen,codec_proof=proof,profile_environment=profile,cases=[],events=[],plans=PLANS,official_benchmark=False,adopted=False,writer_included=True,scope='Soft8 prefill tile scan with real builder/writer/forward; no model service or quality proof')
 def save():
  t=root/'result.tmp';t.write_text(json.dumps(d,indent=2)+'\n');t.replace(root/'result.json')
 save()
 try:
  with torch.inference_mode():
   for case,(name,initial_length,strided) in enumerate((('prefill2048',2048,False),('chunk2048',4096,True))):
    torch.manual_seed(2026101960+case);length=initial_length;width=(length+15)//16;pages=width
    logical_k=torch.randn(1,width*16,2,128,device='cuda',dtype=torch.bfloat16);logical_v=torch.randn_like(logical_k)
    q=torch.randn(2048,20 if strided else 16,128,device='cuda',dtype=torch.bfloat16)[:,:16]
    cu_cpu=torch.tensor([0,2048],dtype=torch.int32);cu=cu_cpu.cuda();used=torch.tensor([length],device='cuda',dtype=torch.int32)
    table=torch.randperm(pages,device='cuda').int().reshape(1,width)
    layer=SimpleNamespace(_q_scale=torch.ones(1,device='cuda'),_k_scale=torch.ones(1,device='cuda'),_v_scale=torch.ones(1,device='cuda'))
    buffers=(torch.full((64,16,64,128),17.,device='cuda',dtype=torch.float32),torch.full((64,16,64),17.,device='cuda',dtype=torch.float32),torch.full((64,16,64),17.,device='cuda',dtype=torch.float32))
    fixtures=[]
    for index,c in enumerate([None,*PLANS]):
     dtype=torch.bfloat16 if c is None else torch.uint8;raw=torch.full(((pages+2)*(2 if strided else 1),2,16,2,128),17,device='cuda',dtype=dtype)
     kv=raw[::2] if strided else raw;k,v=kv.unbind(1)
     slots,ki,vi=updates(logical_k,logical_v,[2048],[length],table,16)
     common=SimpleNamespace(num_reqs=1,num_actual_tokens=2048,max_query_len=2048,max_seq_len=length,query_start_loc=cu,query_start_loc_cpu=cu_cpu,seq_lens=used,block_table_tensor=table,slot_mapping=slots,causal=True,mm_req_doc_ranges=None)
     builder=object.__new__(backend.TritonAttentionMetadataBuilder);builder.seq_threshold_3D=64;builder.num_par_softmax_segments=64;builder.device=q.device
     builder.softmax_segm_output,builder.softmax_segm_max,builder.softmax_segm_expsum=buffers
     metadata=builder.build(0,common);out_storage=torch.full((2061,20 if strided else 16,128),17.,device='cuda',dtype=q.dtype)
     fixtures.append(dict(index=index,label='BF16' if c is None else f'soft8_{index-1}',config=c,storage=raw,kv=kv,k=k,v=v,impl=make_impl(),metadata=metadata,slots=slots,ki=ki,vi=vi,out_storage=out_storage,out=out_storage[:2048,:16]))
    def invoke(f):
     if f['config'] is not None:route['producer'].prefill_config=f['config']
     f['impl'].do_kv_cache_update(layer,f['ki'],f['vi'],f['kv'],f['slots'])
     f['impl'].forward(layer,q,None,None,f['kv'],f['metadata'],f['out'])
    phases=[]
    for phase in (0,1):
     if phase:logical_k.mul_(.5);logical_v.neg_();q.neg_();length=initial_length-min(7,initial_length-2048);used.fill_(length)
     slots,ki,vi=updates(logical_k,logical_v,[2048],[length],table,16)
     for f in fixtures:
      kk=logical_k if f['config'] is None else quantize(logical_k);vv=logical_v if f['config'] is None else quantize(logical_v)
      f['k'][table.flatten().long()]=kk.reshape(pages,16,2,128);f['v'][table.flatten().long()]=vv.reshape(pages,16,2,128)
      expected_storage=f['storage'].clone();f['slots'].copy_(slots);f['ki'].copy_(ki);f['vi'].copy_(vi)
      poison(f['k'],f['v'],f['slots'],16)
      if not phase:
       for _ in range(3):invoke(f)
       graph=torch.cuda.CUDAGraph()
       with torch.cuda.graph(graph):invoke(f)
       f['graph']=graph
       if f['config'] is not None:
        matching=[r for r in route['producer'].records.values() if not r['IS_3D'] and r['BLOCK_M']==f['config']['BLOCK_M'] and r['tile']==f['config']['TILE_SIZE'] and r['warps']==f['config']['num_warps']]
        assert matching;f['compiled_metadata']=dict(matching[-1])
      rng=torch.cuda.get_rng_state().clone();f['graph'].replay();torch.cuda.synchronize();assert torch.equal(torch.cuda.get_rng_state(),rng)
      assert torch.equal(f['storage'],expected_storage) and bool((f['out_storage'][2048:]==17.).all())
      if strided:assert bool((f['out_storage'][:2048,16:]==17.).all())
      assert all(bool((buffer==17.).all()) for buffer in buffers)
     original_ref=reference(q,logical_k,logical_v,[2048],[length]);soft_k=dequantize(quantize(logical_k));soft_v=dequantize(quantize(logical_v))
     soft_ref=reference(q,soft_k,soft_v,[2048],[length])
     checks=[]
     for f in fixtures:
      check=error(f['out'],original_ref if f['config'] is None else soft_ref);assert check['passed'],(name,f['label'],check)
      checks.append(dict(label=f['label'],reference=check,writer_storage_bitwise=True,padding_preserved=True,scratch_untouched=True,RNG_unchanged=True))
     cpu=cpu_reference(q,logical_k,logical_v,[2048],[length],original_ref);cpu_soft=cpu_reference(q,soft_k,soft_v,[2048],[length],soft_ref)
     phases.append(dict(phase=phase,length=length,query_K_V_changed=bool(phase),length_slots_changed=length!=initial_length,checks=checks,cpu_reference_checks=cpu,cpu_soft_reference_checks=cpu_soft))
    for trial in range(7):
     for f in (fixtures if trial%2==0 else list(reversed(fixtures))):
      start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
      for _ in range(20):f['graph'].replay()
      end.record();end.synchronize();d['events'].append(dict(case=case,label=f['label'],trial=trial,ms=start.elapsed_time(end)/20))
    items=[]
    for f in fixtures:
     vals=[e['ms'] for e in d['events'] if (e['case'],e['label'])==(case,f['label'])]
     items.append(dict(label=f['label'],config=f['config'],median_ms=statistics.median(vals),compiled_metadata=f.get('compiled_metadata')))
    d['cases'].append(dict(case=case,name=name,initial_length=initial_length,strided=strided,phases=phases,variants=items));save()
    print(json.dumps(dict(name=name,variants=items)),flush=True)
    del fixtures,original_ref,soft_ref,expected_storage;torch.cuda.empty_cache()
  assert len(d['cases'])==2 and len(d['events'])==224
  for name,h in frozen.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
  d.update(phase='complete',complete=True)
 except BaseException as error:
  import traceback
  d.update(phase='failed',error=repr(error),traceback=traceback.format_exc())
 finally:signal.alarm(0);save()
 return 0 if d['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
