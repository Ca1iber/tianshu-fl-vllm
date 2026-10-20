# SPDX-License-Identifier: Apache-2.0
"""CPU-only independent full IEEE RNE and Python integer audit of saved GPU words."""
import argparse,hashlib,json,struct
from pathlib import Path
BASE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fp32(x):return struct.unpack('f',struct.pack('f',x))[0]
def main():
    import torch
    torch.set_num_threads(8)
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();root=args.root
    state=json.loads((root/'result.json').read_text());assert state['complete'] and state['phase']=='complete' and len(state['cases'])==5
    for path,digest in state['frozen_sources'].items():assert sha(path)==sha(root/'source_snapshot'/Path(path).relative_to('/'))==digest
    count_q=count_i=count_o=0
    for case in state['cases']:
        m,t,d=(case[k] for k in ('M','T','D'));assert t in (32,64,128) and d==128 and len(case['phases'])==3
        assert set(case['compiled'])=={'False','True'} and all(row['cache_hash'] and row['binary_sha256'] and row['warps']==2 for row in case['compiled'].values())
        for phase in case['phases']:
            path=root/phase['file'];assert sha(path)==phase['sha256'];data=torch.load(path,map_location='cpu',weights_only=True)
            p=data['p'];v=data['v'];s=data['value_scales'];assert p.shape==(m,t) and v.shape==(t,d) and s.shape==(t,)
            scales=[];quant=[];vrows=v.tolist()
            for row in p.tolist():
                weighted=[fp32(a*b) for a,b in zip(row,s.tolist())];scale=fp32(max(max(map(abs,weighted)),fp32(1.e-30))/127.)
                scales.append(scale);quant.append([max(0,min(127,round(fp32(value/scale)))) for value in weighted])
            integers=[[sum(row[k]*vrows[k][column] for k in range(t)) for column in range(d)] for row in quant]
            expected_integer=torch.tensor(integers,dtype=torch.int64);expected_scale=torch.tensor(scales,dtype=torch.float32);expected_q=torch.tensor(quant,dtype=torch.int8)
            expected=torch.tensor([[fp32(value*scale) for value in row] for row,scale in zip(integers,scales)],dtype=torch.float32)
            assert bool(expected_integer.abs().max()<=t*127*127)
            for diag in ('False','True'):
                actual=data['actual'][diag];assert actual['out'].shape==(m+3,d) and actual['out'].dtype==torch.float32
                assert torch.equal(actual['out'][:m].view(torch.int32),expected.view(torch.int32)) and bool((actual['out'][m:]==17).all())
                if diag=='True':
                    assert torch.equal(actual['integer'][:m].long(),expected_integer) and torch.equal(actual['quantized'][:m],expected_q)
                    assert torch.equal(actual['scales'][:m].view(torch.int32),expected_scale.view(torch.int32))
                    assert all(bool((actual[k][m:]==17).all()) for k in ('integer','quantized','scales'))
                else:assert all(bool((actual[k]==17).all()) for k in ('integer','quantized','scales'))
            for actual,expected in zip(data['inputs_after'],(p,v,s)):
                size=expected.shape[0];assert torch.equal(actual[:size],expected) and bool((actual[size:]==17).all())
            for prefix in ('CPU','GPU'):assert torch.equal(data[prefix+'_RNG_before'],data[prefix+'_RNG_after'])
            count_q+=m*t;count_i+=m*d;count_o+=2*m*d
    gate=dict(complete=True,independently_CPU_rechecked=True,actual_configurations=5,changed_input_phases=15,
        actual_Pq_RNE_elements=count_q,actual_INT32_vs_independent_Python_INT64_outputs=count_i,both_actual_FP32_output_words=count_o,
        all_guards_inputs_and_RNG_unchanged=True,full_attention_not_run=True,Math_not_run=True,adopted=False,performance_not_measured=True,
        raw_root=str(root),raw_result_sha256=sha(root/'result.json'),frozen_sources=state['frozen_sources'],verifier_sha256=sha(__file__))
    (BASE/'PRIMITIVE_GPU_GATE_32g.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps({k:v for k,v in gate.items() if k!='frozen_sources'}))
if __name__=='__main__':main()
