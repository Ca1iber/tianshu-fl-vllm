# SPDX-License-Identifier: Apache-2.0
"""CPU layout/rounding test; does not execute or qualify the Triton kernel."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'tianshu_k8v16_pipeline_20261019'))
from value_planes import value_plane,clear_cache


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def encode(data):
    bits=data.contiguous().view(torch.uint16).numpy().astype(np.uint32)
    mag=bits&32767;rounded=(mag+7+((mag>>4)&1))>>4
    packed=np.where(rounded>=936,np.minimum(np.maximum(rounded.astype(np.int64)-928,0),127),0)
    return torch.from_numpy((packed|((bits&32768)>>8)).astype(np.uint8))

def scalar_rotation(data,positions,cache):
    values=data.reshape(len(positions),-1,128).float()
    c=cache[positions,:64].float()[:,None,:];s=cache[positions,64:].float()[:,None,:]
    a,b=values[...,:64],values[...,64:]
    p1=(a*c).bfloat16().float();p2=(b*s).bfloat16().float()
    p3=(b*c).bfloat16().float();p4=(a*s).bfloat16().float()
    return torch.cat(((p1-p2).bfloat16(),(p3+p4).bfloat16()),dim=-1).reshape(data.shape)


def main():
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
    generator=torch.Generator().manual_seed(2026102042)
    # Include the model's last legal position, separate/strided input views,
    # negative padding slots, scattered blocks and final token of a page.
    cache=torch.randn((131072,128),generator=generator,dtype=torch.float32).bfloat16()
    cases=[]
    for n in (1,8,32,40,41,63,64):
      for stride in (2560,2688):
        clear_cache()
        qkv=torch.randn((n,stride),generator=generator,dtype=torch.float32).bfloat16()
        q,k,v=qkv[:,:2048],qkv[:,2048:2304],qkv[:,2304:2560]
        pos=torch.randint(0,131072,(n,),generator=generator,dtype=torch.int64);pos[-1]=131071
        slots=torch.arange(n,dtype=torch.int64)*17+15;slots[::5]=-1
        blocks=int(max(1,slots.max().item()//16+1))
        raw=torch.full((blocks*12288+256,),173,dtype=torch.uint8)
        kc=torch.as_strided(raw,(blocks,16,2,128),(12288,256,128,1),storage_offset=128)
        rv=torch.as_strided(raw,(blocks,16,2,128),(12288,256,128,1),storage_offset=128+4096)
        vc=value_plane(rv)
        assert vc.stride()==(6144,256,128,1)
        original=raw.clone();original_input=qkv.clone()
        oq,ok=RotaryEmbedding.forward_static(pos,q,k,128,128,cache,True)
        assert torch.equal(oq,scalar_rotation(q,pos,cache)) and torch.equal(ok,scalar_rotation(k,pos,cache))
        expected=original.numpy().copy()
        ek=encode(ok.reshape(n,2,128)).numpy()
        ev=v.contiguous().view(torch.uint8).numpy().reshape(n,512)
        for r,slot in enumerate(slots.tolist()):
          if slot<0:continue
          b,t=divmod(slot,16)
          start=128+b*12288+t*256
          expected[start:start+256]=ek[r].reshape(-1)
          start=128+b*12288+4096+t*512
          expected[start:start+512]=ev[r]
          kc[b,t]=encode(ok.reshape(n,2,128))[r]
          vc[b,t]=v.reshape(n,2,128)[r]
        assert np.array_equal(raw.numpy(),expected)
        assert torch.equal(qkv,original_input)
        cases.append(dict(n=n,input_row_stride=stride,max_position=int(pos.max()),valid_slots=int((slots>=0).sum()),full_storage_bytes=raw.numel(),all_storage_and_guard_bytes_match=True,native_RoPE_BF16_intermediate_match=True))
    result=dict(complete=True,CPU_only=True,Triton_not_executed=True,not_a_service_or_quality_result=True,
        cases=cases,count=len(cases),sources={str(p):digest(p) for p in (BASE/'kernel.py',BASE/'check_cpu.py')},
        reference='actual RotaryEmbedding.forward_static, selected when current custom_ops none',
        native_forward_source=__import__('inspect').getsourcefile(RotaryEmbedding.forward_static))
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
if __name__=='__main__':main()
