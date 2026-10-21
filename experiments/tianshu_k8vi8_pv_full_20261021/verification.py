# SPDX-License-Identifier: Apache-2.0
"""Full GPU declared-algorithm checks and independently reconstructible CPU inputs."""
import numpy as np
import torch
from plans import PLANS,ALGORITHM_ATOL,ALGORITHM_RTOL,ALGORITHM_RELATIVE_L2
from reference_numpy import Oracle
import reference_torch

def compare(actual,reference,algorithm=False):
    actual=actual.float();reference=reference.to(actual.device).float()
    diff=(actual-reference).abs()
    atol,rtol=(ALGORITHM_ATOL,ALGORITHM_RTOL) if algorithm else (.02,.01)
    relative=float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(reference).clamp_min(1.e-12))
    bad=diff>atol+rtol*reference.abs()
    passed=bool(torch.isfinite(actual).all() and torch.isfinite(reference).all() and not bad.any())
    if algorithm:passed=passed and relative<=ALGORITHM_RELATIVE_L2
    return dict(passed=passed,failed_elements=int(bad.sum()),max_abs=float(diff.max()),relative_L2=relative,atol=atol,rtol=rtol,algorithm=algorithm)

class Context:
    def __init__(self,q,k,v,lengths,segments,encode,decode,quant,case,phase):
        self.n=q.shape[0];self.lengths=list(lengths);self.segments=segments;self.gpu_cache={};self.full_outputs={}
        self.selected=sorted(set((0,self.n//2,self.n-1)))
        maximum=max(lengths[i] for i in self.selected)
        self.data=dict(batch=self.n,segments=segments,case=case,phase=phase,indices=self.selected,
            lengths=[lengths[i] for i in self.selected],q=q[self.selected].cpu(),k=k[self.selected,:maximum].cpu(),v=v[self.selected,:maximum].cpu(),actual_outputs={})
        self.oracle=Oracle(self.data)
        keys=decode(encode(k));self.vq,self.vs=quant(v)
        self.score=reference_torch.scores(q,keys,lengths)
        self.current=reference_torch.ideal(self.score,v)
        self.v8=reference_torch.ideal(self.score,self.vq.float()*self.vs[...,None])
    def validate(self,f,record):
        label=f['mode'];plan=PLANS[label];active=bool(record['PV_INT8'])
        assert active==(plan['pv'] and self.n<=64) and record['IS_3D']==(self.n<=64)
        ideal=self.current if label=='K8V16' else self.v8
        if active:
            tile=record['tile']
            if tile not in self.gpu_cache:self.gpu_cache[tile]=reference_torch.integer_pv(self.score,self.vq,self.vs,self.lengths,tile,self.segments)
            reference=self.gpu_cache[tile]
        else:reference=ideal
        observed=f['output'][:self.n]
        full=compare(observed,reference,active);assert full['passed'],('declared_algorithm',label,full)
        ideal_check=compare(observed,ideal)
        if not active:assert ideal_check['passed']
        selected_output=observed[self.selected].cpu();self.data['actual_outputs'][label]=selected_output
        self.full_outputs[label]=observed.clone()
        checks=[];precision=[]
        for local,sequence in enumerate(self.selected):
            for head in (0,7,8,15):
                actual=selected_output[local,head].float()
                ideal_cpu=self.oracle.ideal(local,head,'current' if label=='K8V16' else 'v8')
                if active:
                    answer,bound=self.oracle.integer_pv(local,head,record['tile'],self.segments)
                    own=compare(actual,torch.from_numpy(answer),True);assert own['passed'],('CPU_integer_attention',label,sequence,head,own)
                    error=np.abs(actual.numpy().astype(np.float64)-ideal_cpu)
                    allowed=bound+.004*np.abs(answer.astype(np.float64))+5.e-4
                    bound_passed=bool(np.all(error<=allowed));assert bound_passed,('quantization_bound',label,sequence,head,float(np.max(error-allowed)))
                else:
                    answer=ideal_cpu;bound_passed=True
                    own=compare(actual,torch.from_numpy(answer));assert own['passed'],('CPU_native_attention',label,sequence,head,own)
                own.update(sequence=sequence,head=head,quantization_bound_passed=bound_passed,ideal_reference=compare(actual,torch.from_numpy(ideal_cpu)))
                checks.append(own)
                original=self.oracle.ideal(local,head,'original');delta=actual.numpy().astype(np.float64)-original
                precision.append(dict(sequence=sequence,head=head,relative_L2=float(np.linalg.norm(delta)/max(np.linalg.norm(original),1.e-8)),max_abs=float(np.max(np.abs(delta)))))
        return dict(GPU_reference=full,ideal_GPU_reference=ideal_check,CPU_references=checks,original_BF16_precision_observations=precision,
            reference_kind='declared_tiled_integer_PV' if active else 'ideal_current_storage_attention',PV_INT8_active=active)
    def finish(self):
        assert torch.equal(self.full_outputs['PVFalse'],self.full_outputs['V8Native'])
        fallback=[]
        if self.n>64:
            for label,plan in PLANS.items():
                if plan['pv']:
                    assert torch.equal(self.full_outputs[label],self.full_outputs['V8Native']);fallback.append(label)
        return dict(default_false_bitwise_native_V8=True,actual_2D_fallback_bitwise_labels=fallback)
