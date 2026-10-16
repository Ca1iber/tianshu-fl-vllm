# SPDX-License-Identifier: Apache-2.0
"""Audit software finite8 KV fixture records without claiming model quality."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
from codec_variant import render_source,decode_scalar,check_cpu,legacy


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def elf_text_sections(raw):
    if raw[:4]!=b'\x7fELF':return {}
    endian='<' if raw[5]==1 else '>'
    wide=raw[4]==2;assert raw[4] in (1,2) and raw[5] in (1,2)
    offset=struct.unpack_from(endian+('Q' if wide else 'I'),raw,40 if wide else 32)[0]
    step,count,names_index=struct.unpack_from(endian+'HHH',raw,58 if wide else 46)
    fmt=endian+('IIQQQQIIQQ' if wide else 'IIIIIIIIII')
    headers=[struct.unpack_from(fmt,raw,offset+i*step) for i in range(count)]
    name_header=headers[names_index];names=raw[name_header[4]:name_header[4]+name_header[5]]
    result={}
    for h in headers:
        name=names[h[0]:].split(b'\0',1)[0].decode()
        if name.startswith('.text') and h[5]:result[name]=hashlib.sha256(raw[h[4]:h[4]+h[5]]).hexdigest()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root.resolve()
    state=json.loads((root/'result.json').read_text());assert state['phase']=='complete' and state['complete']
    assert not state['adopted'] and not state['official_benchmark'] and state['full_model_and_Math_not_run']
    assert state['manual_quantization_not_native_writer'] and state['software_custom_finite_E4M3'] and state['not_native_FP8_hardware']
    repo=Path(__file__).resolve().parents[2]
    for name,sha in state['source_snapshots'].items():
        assert digest(root/'source_snapshot'/Path(name).relative_to('/'))==sha,name
        if Path(name).is_relative_to('/workspace/vllm-plugin-FL'):
            assert digest(repo/Path(name).relative_to('/workspace/vllm-plugin-FL'))==sha,name
    original=next(root.glob('decode_coherent_records/cache_generated/*/generated_cache_policy.py'))
    generated=root/'soft8_generated/generated_soft8_u32.py';text,derivation=render_source(original.read_bytes());assert generated.read_text()==text
    for k,v in derivation.items():assert state['derivation'][k]==v
    old_text,old_record=legacy.render_source(original.read_bytes())
    assert (root/'legacy_soft8_generated/generated_soft8.py').read_text()==old_text
    for k,v in old_record.items():assert state['legacy_derivation'][k]==v
    assert state['CPU_gate']==check_cpu()
    actual=state['all256_actual_GPU_codes'];expected=[decode_scalar(c) for c in range(256)]
    assert state['all256_GPU_decode_equal_CPU_reference'] and actual==expected
    assert all(math.copysign(1.,a)==math.copysign(1.,b) for a,b in zip(actual,expected))
    plans={'BF16':(16,1),'Soft8_u16_t16_w1_s8':(16,1),'Soft8_u32_t16_w1_s8':(16,1),'Soft8_u32_t16_w2_s8':(16,2),'Soft8_u32_t32_w2_s8':(32,2)}
    assert [(c['batch'],c['context']) for c in state['cases']]==[(64,4096),(64,5120)]
    events=checks=CPU_checks=0;timings=[];approximations=[];assembly_files={};text_sections={}
    for c in state['cases']:
        assert c['lengths']==[c['context']-i*37%257 for i in range(64)]
        assert not c['compile_failures'] and len(c['quantization'])==2
        for q in c['quantization']:
            assert q['K_scale']==q['V_scale']==1/16 and q['K_clipped']==q['V_clipped']==0
            count=64*((c['context']+15)//16)*16*2*128;assert q['elements_per_K_or_V']==count
            assert 0<=q['K_flushed']<=count and 0<=q['V_flushed']<=count
        rows=c['results'];assert len(rows)==5 and {r['name'] for r in rows}==plans.keys()
        before=next(r['median_ms'] for r in rows if r['name']=='BF16')
        for r in rows:
            m=r['metadata'];tile,warps=plans[r['name']]
            assert len(r['raw_ms'])==7 and all(math.isfinite(t) and t>0 for t in r['raw_ms'])
            assert statistics.median(r['raw_ms'])==r['median_ms']
            assert abs(r['time_change_percent']-100*(r['median_ms']/before-1))<1e-12
            assert m['grid']==[96,2,8] and m['tile']==tile and m['warps']==m['compiled_warps']==warps
            assert m['quant_mode']==0 and m['key_dtype']==('torch.bfloat16' if r['name']=='BF16' else 'torch.uint8')
            assert m['n_regs']>0 and m['n_spills']>=0
            assert m['codec_intermediates_bits']==(None if r['name']=='BF16' else 16 if '_u16_' in r['name'] else 32)
            remote_root=Path(state['derivation']['generated_source']).parent.parent
            assert {'asm','cubin'}<=m['assembly'].keys()
            for kind,info in m['assembly'].items():
                path=root/Path(info['path']).relative_to(remote_root)
                assert path.stat().st_size==info['bytes'] and digest(path)==info['sha256']
                assembly_files[str(path.relative_to(root))]=info['sha256']
                if kind=='cubin':text_sections[f"{c['context']}/{r['name']}"]=elf_text_sections(path.read_bytes())

            assert r['segments']==8 and not r['scale_folded'] and r['key_strides']==[8192,256,128,1]
            timings.append(dict(batch=64,context=c['context'],**r));events+=7
        assert [p['phase'] for p in c['phases']]==[0,1]
        for p in c['phases']:
            assert p['same_format_wordwidth_outputs_bitwise_equal']
            assert p['outputs_cleared_before_replay'] and len(p['checks'])==5 and {x['name'] for x in p['checks']}==plans.keys()
            for key in ('CPU_BF16_reference','CPU_dequantized_reference'):
                assert len(p[key])==12 and all(x['passed'] and x['failed_elements']==0 for x in p[key]);CPU_checks+=12
            for x in p['checks']:
                a=x['against_own_FP32'];assert a['passed'] and a['failed_elements']==0 and a['atol']==.02 and a['rtol']==.01 and x['tails_preserved']
                assert all(math.isfinite(v) and v>=0 for v in x['against_BF16_FP32'].values());checks+=1
                approximations.append(dict(context=c['context'],phase=p['phase'],name=x['name'],**x['against_BF16_FP32']))
    assert events==70 and checks==20 and CPU_checks==96
    audit=dict(complete=True,source_snapshots_verified=len(state['source_snapshots']),generated_source_reconstructed=True,
        producer_and_reducer_AST_unchanged=True,same_format_wordwidth_outputs_bitwise_equal_all4_phases=True,all_compiled_assembly_artifacts_SHA_verified=assembly_files,machine_text_sections=text_sections,u16_u32_machine_text_identical={str(context):text_sections[f"{context}/Soft8_u16_t16_w1_s8"]==text_sections[f"{context}/Soft8_u32_t16_w1_s8"] if text_sections[f"{context}/Soft8_u16_t16_w1_s8"] else None for context in (4096,5120)},all65280_CPU_encodings_checked=True,all256_actual_GPU_decodes_equal_independent_CPU=True,
        all70_event_groups_recomputed=True,four_changed_input_phase_records_checked=True,own_FP32_checks=checks,CPU_head_checks=CPU_checks,
        full_GPU_tensor_values_not_serialized=True,real_writer_allocator_model_Math_not_run=True,native_FP8_hardware_claimed=False,
        adopted=False,timings=timings,approximations=approximations)
    (root/'local_download_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('timings','approximations','all_compiled_assembly_artifacts_SHA_verified','machine_text_sections')}))


if __name__=='__main__':main()
