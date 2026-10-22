# SPDX-License-Identifier: Apache-2.0
"""Read-only all-layer weight quantization survey; does not infer model quality."""
import argparse,csv,hashlib,json,math,os,struct,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from formats_cpu import FORMATS,bf16_from_words,check_formats,quantize
BASE=Path(__file__).resolve().parent
MODEL=Path('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(p,o):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(o,indent=2)+'\n');t.replace(p)

def main():
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args();root=args.output.resolve()
    assert root.is_relative_to('/workspace/benchmark-results') and not root.exists();root.mkdir()
    sources={str(p):sha(p) for p in [BASE/'formats_cpu.py',Path(__file__),BASE/'PLAN_32g.md']}
    state=dict(complete=False,phase='checking_formats',CPU_only=True,GPU_frameworks_not_imported=True,model_executed=False,model_quality_not_proven=True,performance_not_measured=True,adopted=False,runner_pid=os.getpid(),started_at_utc=datetime.now(timezone.utc).isoformat(),sources=sources,processed_tensors=0)
    def checkpoint():save(root/'state.json',state)
    checkpoint()
    try:
        state['format_checks']=check_formats();state['phase']='hashing_checkpoint';checkpoint()
        before=sha(MODEL);state['checkpoint_sha256_before']=before
        with MODEL.open('rb') as f:
            size=struct.unpack('<Q',f.read(8))[0];meta=json.loads(f.read(size))
        names=sorted(k for k in meta if k=='lm_head.weight' or (k.startswith('model.layers.') and k.endswith(('.self_attn.q_proj.weight','.self_attn.k_proj.weight','.self_attn.v_proj.weight','.self_attn.o_proj.weight','.mlp.gate_proj.weight','.mlp.up_proj.weight','.mlp.down_proj.weight'))))
        assert len(names)==42*7+1,len(names)
        rows=[];state.update(phase='surveying_weights',expected_tensors=len(names));checkpoint()
        for index,name in enumerate(names):
            desc=meta[name];assert desc['dtype']=='BF16' and len(desc['shape'])==2
            n,k=desc['shape'];lo,hi=desc['data_offsets'];assert hi-lo==n*k*2
            words=np.memmap(MODEL,mode='r',dtype='<u2',offset=8+size+lo,shape=(n,k))
            stats={label:dict(tensor=name,role=('lm_head' if name=='lm_head.weight' else name.split('.')[-2]),layer=(-1 if name=='lm_head.weight' else int(name.split('.')[2])),format=label,N=n,K=k,elements=n*k,weight_square_sum=0.,error_square_sum=0.,max_abs_error=0.,clipped_codes=0,original_bytes=n*k*2,packed_bytes=(n*k if bits==8 else k*((n+1)//2)),scale_bytes=n*((k+(group or k)-1)//(group or k))*4,zero_bytes=(n*((k+group-1)//group) if affine else 0)) for label,bits,group,affine in FORMATS}
            h=hashlib.sha256()
            for start in range(0,n,128):
                block=np.asarray(words[start:start+128]);h.update(block.tobytes());w=bf16_from_words(block)
                norm=float(np.sum(w.astype(np.float64)**2))
                for label,bits,group,affine in FORMATS:
                    q,s,z,d,clipped=quantize(w,bits,group,affine)
                    err=d.astype(np.float64)-w.astype(np.float64);st=stats[label]
                    st['weight_square_sum']+=norm;st['error_square_sum']+=float(np.sum(err*err));st['max_abs_error']=max(st['max_abs_error'],float(np.max(np.abs(err))));st['clipped_codes']+=clipped
            rawsha=h.hexdigest()
            for st in stats.values():
                st.update(original_BF16_tensor_sha256=rawsha,NRMSE=math.sqrt(st['error_square_sum']/st['weight_square_sum']) if st['weight_square_sum'] else 0.,payload_bytes=st['packed_bytes']+st['scale_bytes']+st['zero_bytes'])
                st['payload_vs_BF16']=st['payload_bytes']/st['original_bytes'];rows.append(st)
            del words
            state.update(processed_tensors=index+1,last_tensor=name);checkpoint()
            if (index+1)%7==0 or index+1==len(names):print(json.dumps(dict(processed_tensors=index+1,last_tensor=name)),flush=True)
        after=sha(MODEL);assert before==after
        assert all(sha(p)==expected for p,expected in sources.items())
        assert not any(name.split('.')[0] in {'torch','triton','vllm','flag_gems'} for name in sys.modules)
        roles={}
        for row in rows:
            key=row['role']+'/'+row['format'];out=roles.setdefault(key,dict(role=row['role'],format=row['format'],tensors=0,elements=0,error_square_sum=0.,weight_square_sum=0.,payload_bytes=0,original_bytes=0,max_layer_NRMSE=0.))
            for field in ('elements','error_square_sum','weight_square_sum','payload_bytes','original_bytes'):out[field]+=row[field]
            out['tensors']+=1;out['max_layer_NRMSE']=max(out['max_layer_NRMSE'],row['NRMSE'])
        for row in roles.values():row['NRMSE']=math.sqrt(row['error_square_sum']/row['weight_square_sum']);row['payload_vs_BF16']=row['payload_bytes']/row['original_bytes']
        with (BASE/'WEIGHT_ERROR_32g.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
        with (BASE/'ROLE_SUMMARY_32g.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(next(iter(roles.values()))),lineterminator='\n');writer.writeheader();writer.writerows(roles.values())
        state.update(complete=True,phase='complete',checkpoint_sha256_after=after,checkpoint_unchanged=True,verified_tensors=len(names),format_rows=len(rows),role_format_rows=len(roles),ended_at_utc=datetime.now(timezone.utc).isoformat(),weight_error_CSV_sha256=sha(BASE/'WEIGHT_ERROR_32g.csv'),role_summary_CSV_sha256=sha(BASE/'ROLE_SUMMARY_32g.csv'),no_model_accuracy_or_speed_claim=True)
        save(BASE/'CPU_SURVEY_GATE_32g.json',state)
    except BaseException as error:state.update(phase='failed',error=repr(error));raise
    finally:checkpoint();save(root/'result.json',state)
if __name__=='__main__':main()
