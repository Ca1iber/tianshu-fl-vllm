# SPDX-License-Identifier: Apache-2.0
import csv,importlib.util,json,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
EXP=BASE.parent
ROOT=Path('/dev/shm/tianshu_W4_affine_fit_CPU_20261024')
spec=importlib.util.spec_from_file_location('_affine_fit_original_component',EXP/'tianshu_W4_attention64_20261024/common.py')
c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)

def fitted(words):
    w=c.f32(words).reshape(2048,16,128)
    original_q,original_s,original_z,_=c.quant(words,'affine')
    q=original_q.T.reshape(2048,16,128).copy()
    s=original_s.T.copy();z=original_z.T.copy()
    def errors(q,s,z):
        reconstructed=c.f32(c.bf16(q.astype(np.float32)*c.f32(s)[...,None]+c.f32(z)[...,None]))
        return np.sum((reconstructed.astype(np.float64)-w)**2,axis=2)
    best_q=q.copy();best_s=s.copy();best_z=z.copy()
    best_error=errors(q,s,z);initial=best_error.copy();history=[]
    wf=w.astype(np.float64);sw=wf.sum(2)
    for iteration in range(6):
        qf=q.astype(np.float64);sq=qf.sum(2)
        denominator=(qf*qf).sum(2)-sq*sq/128
        covariance=(qf*wf).sum(2)-sq*sw/128
        proposed=np.where(denominator>1e-12,covariance/np.maximum(denominator,1e-12),c.f32(s))
        proposed=np.maximum(proposed,1e-8)
        s=c.bf16(proposed.astype(np.float32))
        z=c.bf16(((sw-proposed*sq)/128).astype(np.float32))
        q=np.clip(np.rint((w-c.f32(z)[...,None])/c.f32(s)[...,None]),0,15).astype(np.uint8)
        error=errors(q,s,z);better=error<best_error
        best_q[better]=q[better];best_s[better]=s[better];best_z[better]=z[better];best_error[better]=error[better]
        history.append(dict(iteration=iteration+1,improved_groups=int(better.sum()),total_MSE=float(best_error.sum())))
    assert np.all(best_error<=initial) and np.isfinite(best_error).all()
    packed=c.pack(best_q.reshape(2048,2048).T.copy())
    assert np.array_equal(c.unpack(packed),best_q.reshape(2048,2048).T)
    return packed,best_s.T.copy(),best_z.T.copy(),float(initial.sum()),float(best_error.sum()),history

def main():
    assert not ROOT.exists();ROOT.mkdir()
    checkpoint=c.sha(c.MODEL)
    assert checkpoint=='14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c'
    source=EXP/'tianshu_W4_attention64_20261024/CPU_GATE_32g.json'
    original={b['layer']:b for b in json.loads(source.read_text())['banks'] if b['role']=='O'}
    sources={str(p):c.sha(p) for p in (Path(__file__),BASE/'PLAN_32g.md',Path(c.__file__),source)}
    files={};rows=[];projection=[]
    for layer in range(42):
        words=c.original(layer,'O');assert c.ahash(words)==original[layer]['original_sha256']
        packed,s,z,initial,error,history=fitted(words)
        target=ROOT/f'O_L{layer}_fitted.npz';np.savez(target,w=packed,s=s,z=z)
        files[str(target)]=c.sha(target)
        norm=float(np.sum(c.f32(words).astype(np.float64)**2))
        rows.append(dict(layer=layer,original_sha256=c.ahash(words),initial_MSE=initial,fitted_MSE=error,
            initial_relative_L2=(initial/norm)**.5,fitted_relative_L2=(error/norm)**.5,
            error_MSE_change_percent=(error/initial-1)*100,packed_sha256=c.ahash(packed),scale_sha256=c.ahash(s),zero_sha256=c.ahash(z),history=history))
        if layer in (0,20,41):
            initial_q,initial_s,initial_z,_=c.quant(words,'affine')
            initial_weight=c.f32(c.dequant(initial_q,initial_s,initial_z)).astype(np.float64)
            fitted_weight=c.f32(c.dequant(c.unpack(packed),s,z)).astype(np.float64)
            actual=c.f32(words).astype(np.float64).T
            for phase in range(3):
                x=c.f32(c.inputs(layer,'O',phase)[:8]).astype(np.float64)
                reference=x@actual
                for name,weight in [('affine_minmax',initial_weight),('affine_fitted',fitted_weight)]:
                    out=x@weight;relative=float(np.linalg.norm(out-reference)/np.linalg.norm(reference))
                    projection.append(dict(layer=layer,phase=phase,variant=name,full_FP64_cells=int(out.size),relative_L2=relative))
        print(json.dumps({k:v for k,v in rows[-1].items() if k in ('layer','initial_relative_L2','fitted_relative_L2','error_MSE_change_percent')}),flush=True)
    assert all(c.sha(p)==h for p,h in sources.items()) and c.sha(c.MODEL)==checkpoint
    assert not any(n.split('.')[0] in ('torch','triton','vllm','flag_gems') for n in sys.modules)
    result=dict(complete=True,CPU_only=True,checkpoint_unchanged=True,checkpoint_sha256=checkpoint,
        original_weights_only_no_activation_or_eval_calibration=True,all42_full_pack_roundtrip=True,
        every_group_actual_BF16_MSE_nonincreasing=True,sources=sources,files=files,banks=rows,projection=projection,
        GPU_qualified=False,model_quality_qualified=False,adopted=False)
    (ROOT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    summary={k:v for k,v in result.items() if k not in ('banks','files','projection')}
    summary.update(raw_root=str(ROOT),result_sha256=c.sha(ROOT/'result.json'),
        total_initial_MSE=sum(r['initial_MSE'] for r in rows),total_fitted_MSE=sum(r['fitted_MSE'] for r in rows),
        mean_initial_projection_L2=float(np.mean([r['relative_L2'] for r in projection if r['variant']=='affine_minmax'])),
        mean_fitted_projection_L2=float(np.mean([r['relative_L2'] for r in projection if r['variant']=='affine_fitted'])))
    (BASE/'CPU_GATE_32g.json').write_text(json.dumps(summary,indent=2)+'\n')
    for name,data,fields in [('KEY_WEIGHT_ERRORS_32g.csv',rows,['layer','initial_relative_L2','fitted_relative_L2','error_MSE_change_percent']),
                             ('KEY_PROJECTION_ERRORS_32g.csv',projection,list(projection[0]))]:
        with (BASE/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(data)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
