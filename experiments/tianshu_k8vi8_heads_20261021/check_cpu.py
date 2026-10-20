# SPDX-License-Identifier: Apache-2.0
"""Check real native page accounting and the exact limited reader source edit."""
import hashlib,inspect,json,py_compile
from pathlib import Path
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache
from layout import MODES,allocate
import source
BASE=Path(__file__).resolve().parent
CV=source.SOURCE
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    text,proof=source.render(CV.read_bytes());compile(text,'<generated-reader>','exec')
    checks=[]
    for mode in MODES:
      for strided in (False,True):
        raw,native,k,v,ks,vs,layout=allocate(torch,FullAttentionSpec,_reshape_attention_kv_cache,mode,10,strided,device='cpu')
        assert native.stride(0)==layout['page_bytes']*(2 if strided else 1)
        expected=torch.full_like(raw,17)
        for plane,value,offset in ((k,3,0),(v,5,4096),(ks,7,8192),(vs,11,8320)):
            if plane is None:continue
            plane.fill_(value)
            size=plane.element_size();rowbytes=plane[0].numel()*size
            encoded=plane[0].contiguous().view(torch.uint8).flatten()
            for row in range(plane.shape[0]):
                start=row*native.stride(0)+offset;expected[start:start+rowbytes]=encoded
            assert torch.equal(raw,expected),(mode,strided,offset)
        checks.append(layout)
    sources={str(p):sha(p) for p in [*BASE.glob('*.py'),*BASE.glob('*.sh'),CV,
        BASE.parent/'tianshu_soft8_mixed_planes_20261019/writer.py',Path(inspect.getsourcefile(FullAttentionSpec)),Path(inspect.getsourcefile(_reshape_attention_kv_cache))]}
    for p in BASE.glob('*.py'):py_compile.compile(str(p),doraise=True)
    result=dict(complete=True,reader_proof=proof,native_layout_checks=checks,sources=sources,
        actual_GPU_writer_reader_not_yet_verified=True,full_engine_not_installed=True,Math_not_run=True,
        physical_K8V16_page_bytes=12288,physical_new_page_bytes=8448,physical_reduction_fraction=(12288-8448)/12288)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,native_layout_checks=len(checks),sources=len(sources),page_bytes=8448)))
if __name__=='__main__':main()
