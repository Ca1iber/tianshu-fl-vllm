# SPDX-License-Identifier: Apache-2.0
"""Revalidate the existing fixed gather through native public and Torch APIs."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-at-utc',required=True)
    a=p.parse_args();root=a.output.resolve();assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir(parents=True)
    remaining=int((datetime.fromisoformat(a.stop_at_utc)-datetime.now(timezone.utc)).total_seconds());assert remaining>300;signal.alarm(min(300,remaining))
    d=dict(phase='preparing',complete=False,runner_pid=os.getpid(),source_snapshots={},torch_api_cases=[],official_benchmark=False,adopted=False)
    def save():
        tmp=root/'result.tmp';tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(root/'result.json')
        print(json.dumps(dict(phase=d['phase'],error=d.get('error'))),flush=True)
    def snapshot(path):
        path=Path(path).resolve();raw=path.read_bytes();d['source_snapshots'][str(path)]=hashlib.sha256(raw).hexdigest()
        target=root/'source_snapshot'/path.relative_to('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    save()
    try:
        hardware=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in hardware;(root/'hardware_before.txt').write_text(hardware)
        import torch
        import vllm_fl
        import flag_gems
        module=importlib.import_module('flag_gems.ops.index')
        source=Path(__file__).resolve().parents[1]/'tianshu_dspark_20261012/fixed_index.py'
        assert hashlib.sha256(source.read_bytes()).hexdigest()==FIXED_SHA
        spec=importlib.util.spec_from_file_location('_ngram_validated_fixed_index',source);fixed=importlib.util.module_from_spec(spec);sys.modules[spec.name]=fixed;spec.loader.exec_module(fixed)
        for path in (Path(__file__),source,Path(module.__file__),Path(flag_gems.__file__)):snapshot(path)
        torch.set_num_threads(8);fixed.run_dispatch_validation(root/'dispatch')
        result=json.loads((root/'dispatch/result.json').read_text());assert result['passed'] and len(result['cases'])==48 and len(result['graph_replays'])==3
        calls=[];fixed.install(module,lambda source,index,out:calls.append(dict(source_shape=list(source.shape),index_rows=index.numel(),dtype=str(source.dtype),index_dtype=str(index.dtype))))
        with torch.inference_mode(),flag_gems.use_gems():
            for width,count in ((1,29),(2048,69),(130560,18)):
                for dtype in (torch.int32,torch.bfloat16):
                    shape=(257,) if width==1 else (129,width)
                    source=torch.randint(-200,201,shape,device='cuda').to(dtype)
                    indices=torch.arange(count,device='cuda',dtype=torch.int32)%source.shape[0];indices[0]=-1
                    before=len(calls);actual=source[indices];expected=source.cpu()[indices.cpu().long()]
                    assert len(calls)>before and torch.equal(actual.cpu(),expected)
                    d['torch_api_cases'].append(dict(width=width,count=count,dtype=str(dtype),actual_Torch_API_fixed_route=True,bitwise_CPU_equal=True))
        for path,h in d['source_snapshots'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
        d.update(phase='complete',complete=True,dispatch_cases=48,changed_graph_checks=3,actual_torch_API_calls=calls,
            scope='Unchanged frozen fixed gather; valid axis0 single1D integer indexing only. Other forms fall back. No model throughput or quality.')
    except Exception as e:
        import traceback
        d.update(phase='failed',error=repr(e),traceback=traceback.format_exc())
    finally:save()
    return 0 if d['complete'] else 1


FIXED_SHA='f48351f7ad1251acb680927d742123f92ea4f3e23505d83dfde6a3b7c713cc06'
if __name__=='__main__':raise SystemExit(main())
