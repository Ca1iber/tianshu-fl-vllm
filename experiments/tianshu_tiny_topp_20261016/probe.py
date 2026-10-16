# SPDX-License-Identifier: Apache-2.0
"""Check the already qualified GEO mask for small rejection-target batches."""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import signal
import statistics
import subprocess
import sys
import textwrap


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True)
    a=p.parse_args();root=a.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir(parents=True)
    remaining=int((datetime.fromisoformat(a.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>120;signal.alarm(min(360,remaining))
    d=dict(phase='preparing',complete=False,runner_pid=os.getpid(),source_snapshots={},cases=[],events=[],compiled={},official_benchmark=False,adopted=False)
    def save():
        tmp=root/'result.tmp';tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(root/'result.json')
        print(json.dumps(dict(phase=d['phase'],cases=len(d['cases']),error=d.get('error'))),flush=True)
    def snapshot(p):
        p=Path(p).resolve();raw=p.read_bytes();d['source_snapshots'][str(p)]=hashlib.sha256(raw).hexdigest();dest=root/'source_snapshot'/p.relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    save()
    try:
        hw=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hw;(root/'hardware_before.txt').write_text(hw)
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tianshu_gemm_topp_geo_20261016'))
        from validation_profile_new import prepare
        _,env,profile,sources=prepare(root,Path('/workspace/benchmark-results/20261016_gemm_topp_geo_service_tianshu'))
        os.environ.clear();os.environ.update(env);runpy.run_path(str(Path(profile['PYTHONPATH'])/'sitecustomize.py'))
        import torch
        import flag_gems
        import vllm_fl
        from vllm.v1.sample.ops import topk_topp_sampler as sampler
        from vllm.v1.sample.ops import topk_topp_triton as native
        geo=sys.modules['_tianshu_topp_geometric_geo'];assert hashlib.sha256(Path(geo.__file__).read_bytes()).hexdigest()=='09156e66b6aa2f4460423ea5b8d70bee043022bd75f0a784417455d0e979479d'
        fixture=Path(__file__).resolve().parents[1]/'tianshu_topp_geometric_20261016/probe_precision.py'
        assert hashlib.sha256(fixture.read_bytes()).hexdigest()==FIXTURE_SHA
        ns=dict(torch=torch);tree=ast.parse(fixture.read_text())
        for name in ('oracle','assess'):
            node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
            exec(compile(textwrap.dedent('\n'.join(fixture.read_text().splitlines()[node.lineno-1:node.end_lineno])),str(fixture),'exec'),ns)
        oracle,assess=ns['oracle'],ns['assess'];torch.set_num_threads(8)
        for source in sources:snapshot(source)
        for path in (Path(__file__),fixture,Path(sampler.__file__),Path(native.__file__),Path(geo.__file__),Path(flag_gems.__file__)):snapshot(path)
        kernel=geo._topk_topp_kernel
        class Router:
            def __getitem__(self,grid):
                def launch(*args,**kw):
                    assert 1<=kw['BATCH_SIZE']<=7 and kw['VOCAB_SIZE']==130560 and kw['TOPP_ENABLED'] and not kw['TOPK_ENABLED']
                    kw.update(BLOCK_SIZE=32768,BLOCK_SIZE_TRUNC=16384,num_warps=16,num_stages=1)
                    result=kernel[grid](*args,**kw)
                    d['compiled'][str(kw['BATCH_SIZE'])]=dict(num_warps=result.metadata.num_warps,num_stages=result.metadata.num_stages,n_regs=getattr(result,'n_regs',None),n_spills=getattr(result,'n_spills',None))
                    return result
                return launch
        geo._topk_topp_kernel=Router();plans=[(1,'normal'),(2,'normal'),(4,'normal'),(7,'normal'),(1,'peaked'),(2,'bf16_peaked'),(4,'ties'),(7,'equal')]
        d['phase']='numerical';save()
        with torch.inference_mode(),flag_gems.use_gems():
            for case,(batch,dist) in enumerate(plans):
                torch.manual_seed(2026102900+case);values=torch.randn(batch,130560,device='cuda')
                if 'peaked' in dist:values.mul_(4)
                if dist.startswith('bf16'):values=values.to(torch.bfloat16).float()
                if dist=='ties':values=values.round()
                if dist=='equal':values.zero_()
                prob=torch.full((batch,),.95,device='cuda');working={m:values.clone() for m in ('native','geo')}
                def fn(mode):
                    working[mode].copy_(values)
                    return (sampler.apply_top_k_top_p if mode=='native' else geo.apply_top_k_top_p_triton)(working[mode],None,prob)
                graphs={}
                for mode in working:
                    for _ in range(3):fn(mode)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):fn(mode)
                    graphs[mode]=graph
                for phase in (0,1):
                    if phase:values.copy_(values.roll(17,dims=1))
                    reference=oracle(values,None,prob);expected=sampler.apply_top_k_top_p(values.clone(),None,prob);checks={}
                    for mode in working:
                        working[mode].fill_(123.);graphs[mode].replay();torch.cuda.synchronize()
                        checks[mode]=assess(working[mode],expected,values,reference,require_strict=True)
                        assert checks[mode]['relative_gate'],(case,phase,mode,checks[mode])
                    rows=[]
                    for trial in range(5):
                        for mode in (('native','geo') if trial%2==0 else ('geo','native')):
                            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
                            for _ in range(10):graphs[mode].replay()
                            end.record();end.synchronize();row=dict(case=case,phase=phase,mode=mode,trial=trial,ms=start.elapsed_time(end)/10);rows.append(row);d['events'].append(row)
                    med={m:statistics.median(r['ms'] for r in rows if r['mode']==m) for m in working}
                    d['cases'].append(dict(case=case,phase=phase,batch=batch,distribution=dist,checks=checks,medians_ms=med,time_delta_percent=100*(med['geo']/med['native']-1),cleared_outputs=True,changed_input=bool(phase)));save()
        for path,h in d['source_snapshots'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
        d.update(phase='complete',complete=True,scope='SmallB1..7 native masking API vs qualified GEO masking body; actual use_gems, no RNG/model/service/quality claim; event timing includes identical input restore.')
    except Exception as e:
        import traceback
        d.update(phase='failed',error=repr(e),traceback=traceback.format_exc())
    finally:save()
    return 0 if d['complete'] else 1


FIXTURE_SHA='3354e14c55aea477ba39497cf5f0378aadc83529f5d7956ceacb6197e9509d2c'
if __name__=='__main__':raise SystemExit(main())
