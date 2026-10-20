# SPDX-License-Identifier: Apache-2.0
"""Check real native page accounting and the exact limited reader source edit."""
import hashlib,inspect,json,os,py_compile
from pathlib import Path
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache
from layout import MODES,allocate
import source
BASE=Path(__file__).resolve().parent
CV=source.SOURCE
REF=BASE.parent/'tianshu_k8vi8_writer_diagnostic_20261021/REFERENCE_RNE_GATE_32g.json'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
    ref=json.loads(REF.read_text());assert ref['complete'] and ref['actual_GPU_FP32_RNE_scale_equals_independent_struct_FP32'] and ref['actual_GPU_quantized_values_equal_CPU']
    assert sha(ref['kernel_source'])==ref['kernel_source_sha256'] and sha(REF.parent/'verify_reference_rne.py')==ref['reference_script_sha256']
    text,proof=source.render(CV.read_bytes());compile(text,'<generated-reader>','exec')
    assert (BASE/'writer.py').read_bytes()==(BASE.parent/'tianshu_k8vi8_heads_v2_20261021/writer.py').read_bytes()
    import struct
    for word in range(65536):
        low=word&255;high=word>>8
        decoded=(low if low<128 else low-256,high if high<128 else high-256)
        assert decoded==struct.unpack('<bb',struct.pack('<H',word))
    assert [j for pair in ((2*i,2*i+1) for i in range(64)) for j in pair]==list(range(128))
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
    sources={str(p):sha(p) for p in [*BASE.glob('*.py'),*BASE.glob('*.sh'),CV,source.PARENT,
        BASE.parent/'tianshu_soft8_mixed_planes_20261019/writer.py',Path(inspect.getsourcefile(FullAttentionSpec)),Path(inspect.getsourcefile(_reshape_attention_kv_cache))]}
    for p in [REF,REF.parent/'verify_reference_rne.py',Path(ref['kernel_source'])]:sources[str(p)]=sha(p)
    for p in BASE.glob('*.py'):py_compile.compile(str(p),doraise=True)
    assert not torch.cuda.is_initialized()
    result=dict(complete=True,CPU_only=True,no_GPU_initialized=True,CPU_CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"],all_65536_V_word_signed_pair_decodings_exact=True,unchanged_actual_writer_produces_K_scales_exact_one=True,explicit_FP32_RNE_reference_verified=True,reader_proof=proof,native_layout_checks=checks,sources=sources,
        actual_GPU_writer_reader_not_yet_verified=True,full_engine_not_installed=True,Math_not_run=True,
        physical_K8V16_page_bytes=12288,physical_new_page_bytes=8448,physical_reduction_fraction=(12288-8448)/12288)
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,native_layout_checks=len(checks),sources=len(sources),page_bytes=8448)))
if __name__=='__main__':main()
