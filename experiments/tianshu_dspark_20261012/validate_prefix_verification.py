# SPDX-License-Identifier: Apache-2.0
"""Check Q2/3/4 target verification through native wrappers, with CPU oracles."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();assert not root.exists();root.mkdir(parents=True)
    deadline=datetime.fromisoformat(json.loads(Path(
        '/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json'
    ).read_text())['stop_at_utc'])
    remaining=int((deadline-datetime.now(timezone.utc)).total_seconds());assert remaining>0
    signal.alarm(min(900,remaining))
    assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
    import torch
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    path=Path(__file__).resolve().parents[1]/'tianshu_native_attention/validate_hybrid.py'
    spec=importlib.util.spec_from_file_location('prefix_fixture',path)
    fixture=importlib.util.module_from_spec(spec);sys.modules[spec.name]=fixture;spec.loader.exec_module(fixture)
    reference=fixture.reference;cpu_checks=[]
    def checked_reference(q,k,v,table,qlens,lengths,causal=True):
        if causal:answer=reference(q,k,v,table,qlens,lengths)
        else:
            parts=[];begin=0
            for req,(ql,kl) in enumerate(zip(qlens,lengths)):
                pages=table[req,:(kl+15)//16].long()
                keys=k[pages].reshape(-1,2,128)[:kl].float().repeat_interleave(8,dim=1)
                values=v[pages].reshape(-1,2,128)[:kl].float().repeat_interleave(8,dim=1)
                query=q[begin:begin+ql].float().transpose(0,1)
                score=query@keys.permute(1,2,0)*128**-.5
                parts.append((score.softmax(-1)@values.transpose(0,1)).transpose(0,1));begin+=ql
            answer=torch.cat(parts)
        qc,kc,vc,tc=q.cpu().float(),k.cpu().float(),v.cpu().float(),table.cpu();offset=0
        for req,(ql,kl) in enumerate(zip(qlens,lengths)):
            if req in {0,len(qlens)-1}:
                pages=tc[req,:(kl+15)//16].long()
                keys=kc[pages].reshape(-1,2,128)[:kl];values=vc[pages].reshape(-1,2,128)[:kl]
                for pos in sorted({0,ql-1}):
                    end=kl-ql+pos+1 if causal else kl
                    for head in (0,8,15):
                        score=keys[:end,head//8]@qc[offset+pos,head]*128**-.5
                        expected=score.softmax(0)@values[:end,head//8]
                        check=fixture.error(answer[offset+pos,head].cpu(),expected)
                        assert check['passed'],check;cpu_checks.append(check)
            offset+=ql
        return answer
    fixture.reference=checked_reference
    original_forward=fixture.backend.TritonAttentionImpl.forward
    cases=[('q2_b1',[2],[4096],False),('q3_b2',[3,3],[4096,4097],True),
           ('q4_b8',[4]*8,[4096]*8,False),('q3_b64',[3]*64,[4096]*64,True),
           ('mixed_q1_q2_q3',[1,2,3]*5,[4097]*15,True)]
    rows=[]
    for idx,(name,qlens,lengths,strided) in enumerate(cases):
        row=fixture.run_case(None,original_forward,name,qlens,lengths,strided,2026101260+idx)
        assert row['candidate_vs_fp32']['passed'] and row['native_vs_fp32']['passed']
        rows.append(row);torch.cuda.empty_cache()
    source=path.read_text();needle='causal=True,mm_req_doc_ranges=None';assert source.count(needle)==1
    clone=root/'noncausal_fixture.py'
    clone.write_text(source.replace(needle,'causal=False,mm_req_doc_ranges=None'))
    spec=importlib.util.spec_from_file_location('noncausal_prefix_fixture',clone)
    other=importlib.util.module_from_spec(spec);sys.modules[spec.name]=other;spec.loader.exec_module(other)
    other.reference=lambda *args:checked_reference(*args,causal=False)
    draft_rows=[]
    for idx,(name,qlens,lengths,strided) in enumerate([
        ('draft_q7_kv7',[7],[7],False),('draft_q7_b8_kv4096',[7]*8,[4096]*8,True),
        ('draft_q7_b64_kv4097',[7]*64,[4097]*64,True)]):
        row=other.run_case(None,original_forward,name,qlens,lengths,strided,2026101270+idx)
        assert row['candidate_vs_fp32']['passed'] and row['native_vs_fp32']['passed']
        draft_rows.append(row);torch.cuda.empty_cache()
    result={'complete':True,'passed':True,'official_benchmark':False,'cases':rows,
        'draft_noncausal_cases':draft_rows,
        'cpu_checks':cpu_checks,'numerical_scope':'Target causal Q2/3/4 and unchanged noncausal Q7 draft; actual metadata+interleaved paged KV, random pages/strides, output sentinels.',
        'timing_scope':'Fixture times the same selected launcher twice; these times are not an optimization comparison.'}
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
