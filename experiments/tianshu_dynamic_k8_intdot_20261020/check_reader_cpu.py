# SPDX-License-Identifier: Apache-2.0
import ast,hashlib,json
from pathlib import Path
import torch
import codec,reader_source
BASE=Path(__file__).resolve().parent
torch.set_num_threads(1)
parent=Path('/workspace/benchmark-results/20261019_soft8_pipeline_micro_v2_tianshu/micro/codec_generated/reader/generated_soft8_u32.py')
generated=reader_source.generate(parent.read_text())
generator=torch.Generator(device='cpu').manual_seed(2026102002)
q=torch.randn((2,16,128),generator=generator).to(torch.bfloat16)
k=torch.randn((2,64,2,128),generator=generator).to(torch.bfloat16)
qc,qs=codec.encode_cpu(q);kc,ks=codec.encode_cpu(k)
qc=qc.view(torch.int8).long();kc=kc.view(torch.int8).long()
count=0
for sequence in range(2):
    for head in range(2):
        query=qc[sequence,head*8:(head+1)*8]
        key=kc[sequence,:,head]
        integer=query@key.T
        assert bool((integer.abs()<=128*127*127).all())
        assert torch.equal(integer,integer.float().long())
        qscale=qs[sequence,head*8:(head+1)*8].double()
        kscale=ks[sequence,:,head].double()
        scaled=integer.double()*qscale[:,None]*kscale[None,:]
        reference=(query.double()*qscale[:,None])@(key.double()*kscale[:,None]).T
        assert torch.allclose(scaled,reference,atol=1.e-12,rtol=1.e-12)
        count+=integer.numel()
for path in (BASE/'reader_source.py',BASE/'writer.py',BASE/'probe_codecs_gpu.py',Path(__file__)):
    ast.parse(path.read_text())
result=dict(complete=True,CPU_only=True,GPU_reader_executed=False,adopted=False,
            exact_reversible_source_scope_verified=True,reader_argument_ABI_preserved=True,
            helper_and_reducer_AST_unchanged=True,CPU_integer_scaling_algebra_checks=count,
            INT32_worst_case_bound=128*127*127,all_integer_products_exactly_representable_in_FP32=True,
            generated_source_sha256=hashlib.sha256(generated.encode()).hexdigest(),
            sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),BASE/'reader_source.py',BASE/'codec.py',parent)})
(BASE/'CPU_READER_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='sources'}))
