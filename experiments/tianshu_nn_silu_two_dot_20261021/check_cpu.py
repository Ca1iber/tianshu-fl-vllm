# SPDX-License-Identifier: Apache-2.0
"""CPU-only NN pair-column mapping, source and qualified tile checks."""
import ast,hashlib,json
from pathlib import Path
import torch
from choices import CASES,configurations,baseline
BASE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    assert not torch.cuda.is_initialized()
    for p in BASE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    official=read(BASE.parent/'tianshu_prefill_int8_official_results_20261021/RUNTIME_BINDING_GATE_32g.json');sources=dict(official['frozen_sources']);assert len(sources)==407
    assert all(sha(p)==h for p,h in sources.items())
    nnpath=str(BASE.parent/'tianshu_int8_weight_nn_20261021/kernel_nn.py');assert nnpath in sources
    selected=read(BASE.parent/'tianshu_int8_nn_gateup_tiles_results_20261021/GPU_GATE_32g.json')
    assert selected['complete'] and not selected['compile_failures']
    assert all(baseline(r['M'])==r['NN_configuration'] for r in selected['best_by_M'])
    mappings=0
    for m,s in CASES:
      for cfg in configurations(m):
        assert cfg['bm']==(32 if m==8 else 64) and cfg['bk']==128 and cfg['stages']==1
        bn=cfg['bn'];assert 6144%bn==0
        gate=[];up=[]
        for block in range(6144//bn):
            cols=block*bn+torch.arange(2*bn)%bn+(torch.arange(2*bn)//bn)*6144
            tile=cols.expand(cfg['bm'],-1).reshape(cfg['bm'],2,bn).permute(0,2,1)
            assert torch.equal(tile[0,:,0],block*bn+torch.arange(bn)) and torch.equal(tile[0,:,1],block*bn+torch.arange(bn)+6144)
            gate.extend(tile[0,:,0].tolist());up.extend(tile[0,:,1].tolist());mappings+=1
        assert gate==list(range(6144)) and up==list(range(6144,12288))
    assert 2048*127*127<2**31
    paths=[*BASE.glob('*.py'),BASE/'run_component.sh',BASE/'PLAN_32g.md',
      BASE.parent/'tianshu_int8_gateup_silu_20261021/probe_gpu.py',
      BASE.parent/'tianshu_int8_nn_gateup_tiles_results_20261021/GPU_GATE_32g.json',
      BASE.parent/'tianshu_dense_w8a8_quality_20261020/PACKED_WEIGHT_SIGNATURE_32g.json',
      Path('/usr/local/lib/python3.12/site-packages/flag_gems/fused/silu_and_mul.py'),
      Path('/usr/local/lib/python3.12/site-packages/triton/language/core.py')]
    sources.update({str(p):sha(p) for p in paths})
    assert not torch.cuda.is_initialized()
    result=dict(complete=True,CPU_only=True,CUDA_not_initialized=True,GPU_not_executed=True,
      qualified407_sources_unchanged=True,current_NN_selected_tiles_verified=True,configuration_shape_cases=16,
      paired_block_mapping_checks=mappings,all_12288_weight_columns_covered_once=True,single_dot_split_order_verified=True,
      signed_INT32_accumulation_bound=2048*127*127,source_count=len(sources),sources=sources)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='sources'}))
if __name__=='__main__':main()
