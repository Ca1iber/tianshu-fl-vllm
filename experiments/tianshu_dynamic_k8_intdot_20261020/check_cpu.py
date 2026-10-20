# SPDX-License-Identifier: Apache-2.0
"""Independent IEEE FP32 arithmetic and BF16 bit-rounding reference; no CUDA use."""
import hashlib,json,math,struct
from pathlib import Path
import torch
import codec
BASE=Path(__file__).resolve().parent
torch.set_num_threads(1)
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def bf16(value):
    bits=struct.unpack('<I',struct.pack('<f',value))[0]
    return ((bits+32767+((bits>>16)&1))>>16)&65535
finite=[bits for bits in range(65536) if (bits&32640)!=32640]
assert len(finite)==65280
signed=torch.tensor([x if x<32768 else x-65536 for x in finite],dtype=torch.int16)
all_values=signed.view(torch.bfloat16)
cases=[('all_finite_grouped',all_values.reshape(-1,128))]
for seed in (20261020,17):
    generator=torch.Generator(device='cpu').manual_seed(seed)
    cases.append((f'all_finite_permuted{seed}',all_values[torch.randperm(len(finite),generator=generator)].reshape(-1,128)))
ties=torch.zeros((4,128),dtype=torch.bfloat16)
ties[0,:10]=torch.tensor([127,-127,.5,-.5,1.5,-1.5,2.5,-2.5,3.5,-3.5],dtype=torch.bfloat16)
ties[1]=torch.finfo(torch.bfloat16).max;ties[2]=-torch.finfo(torch.bfloat16).max
cases.append(('ties_zero_extremes',ties))
records=[];elements=0
for name,x in cases:
    before=x.view(torch.int16).clone()
    codes,scales=codec.encode_cpu(x)
    decoded=codec.decode_cpu(codes,scales)
    expected_codes=[];expected_scales=[];expected_decoded=[]
    for row in x.float().tolist():
        maximum=max(max(abs(value) for value in row),f32(1.e-8))
        scale=f32(maximum/127)
        expected_scales.append(scale)
        for value in row:
            quant=max(-127,min(127,round(f32(value/scale))))
            expected_codes.append(quant&255)
            expected_decoded.append(bf16(f32(quant*scale)))
    assert torch.equal(scales,torch.tensor(expected_scales,dtype=torch.float32))
    assert codes.flatten().tolist()==expected_codes
    assert (decoded.view(torch.int16).to(torch.int32)&65535).flatten().tolist()==expected_decoded
    assert torch.equal(x.view(torch.int16),before)
    assert bool(torch.isfinite(decoded).all()) and bool((codes!=128).all())
    elements+=x.numel()
    delta=decoded.double()-x.double()
    norm=float(x.double().norm())
    records.append(dict(name=name,rows=x.shape[0],elements=x.numel(),codes_exact=True,FP32_scales_exact=True,
                        BF16_decode_bits_exact=True,inputs_unchanged=True,
                        relative_L2=float(delta.norm())/norm if norm else 0,
                        encoded_sha256=hashlib.sha256(codes.numpy().tobytes()+scales.numpy().tobytes()).hexdigest()))
expected=[127,129,0,0,2,254,2,254,4,252]
assert codec.encode_cpu(ties)[0][0,:10].tolist()==expected
occupied=set()
for slot in range(16):
    for head in range(2):
        for dim in range(128):
            key=(slot*2+head)*128+dim
            value=4096+((slot*2+head)*128+dim)*2
            for byte in (key,value,value+1):
                assert byte not in occupied;occupied.add(byte)
        scale=12288+(slot*2+head)*4
        assert scale%4==0
        for byte in range(scale,scale+4):
            assert byte not in occupied;occupied.add(byte)
assert occupied==set(range(codec.PAGE_BYTES))
result=dict(complete=True,CPU_only=True,GPU_format_or_attention_executed=False,model_or_service_executed=False,adopted=False,
            finite_BF16_bit_patterns=65280,total_encoding_scalar_checks=elements,records=records,
            scales_per_token_head='FP32 max(max_abs,1e-8)/127 with FP32 round-nearest',
            codes='signed nearest-even int8, [-127,127], unsigned carrier;128 reserved',
            decode='FP32 signed code*scale then BF16 nearest-even',
            nonfinite_inputs_outside_contract=True,zero_codes_decode_to_positive_zero=True,
            physical_page_bytes=codec.PAGE_BYTES,layout_all_bytes_disjoint_and_exactly_covered=True,
            same_BF16_V_plane_as_current_K8V16=True,new_scale_metadata_bytes_per_page=128,
            does_not_prove_model_accuracy_or_performance=True,
            sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),BASE/'codec.py')})
(BASE/'CPU_GATE_32g.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('sources','records')}))
