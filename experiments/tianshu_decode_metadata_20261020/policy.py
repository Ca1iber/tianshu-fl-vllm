# SPDX-License-Identifier: Apache-2.0
"""Bind the verified integer block to the actual FL runner, preserving all else."""
import atexit,functools,hashlib,importlib.util,json,os
from pathlib import Path
import sys,time
BASE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def install(module,mode,directory):
    assert mode in ('control','candidate')
    source=load('_production_decode_metadata_source',BASE/'source.py')
    kernel=load('_production_decode_metadata_kernel',BASE/'kernel.py')
    cpu=json.loads((BASE/'CPU_GATE_32g.json').read_text());gpu=json.loads((BASE/'GPU_GATE_32g.json').read_text())
    assert cpu['complete'] and gpu['complete'] and gpu['GPU_cases']==14
    for path,digest in cpu['source_sha256'].items():assert sha(path)==digest,path
    for path,digest in gpu['frozen_sources'].items():assert sha(path)==digest,path
    cls=module.ModelRunnerFL;original=cls._prepare_inputs;original_init=cls.__init__
    assert not getattr(cls,'_tianshu_decode_metadata_installed',False)
    actual=Path(original.__code__.co_filename)
    assert sha(actual)==source.NATIVE_SHA and original.__name__=='_prepare_inputs'
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    text,provenance=source.render();generated=directory/'generated_prepare_inputs.py'
    assert hashlib.sha256(text.encode()).hexdigest()==gpu['generated_method_sha256']
    generated.write_text(text)
    binding=dict(installed=True,mode=mode,original_method_source=str(actual),original_method_source_sha256=sha(actual),
        generated_method_source=str(generated),generated_method_source_sha256=sha(generated),
        kernel_source=str(BASE/'kernel.py'),kernel_source_sha256=sha(BASE/'kernel.py'),
        full_other_method_AST_unchanged=True,original_fallback_AST_unchanged=True,provenance=provenance)
    (directory/f'binding_{os.getpid()}.json').write_text(json.dumps(binding,indent=2)+'\n')

    class Controller:
        def __init__(self,owner):
            self.owner=owner;self.last_write=0.
            self.data=dict(mode=mode,calls=0,eligible_calls=0,fallback_calls=0,selected_calls=0,
                fused_launches=0,eligible_by_batch={},selected_by_batch={},actual_kernels={},
                warmup_complete=False,compile_only_no_array_writes=True,RNG_not_used_by_metadata_kernel=True)
        def emit(self,force=False):
            now=time.monotonic()
            if not force and now-self.last_write<5:return
            self.last_write=now;target=directory/f'runtime_{os.getpid()}.json';tmp=target.with_suffix('.tmp')
            tmp.write_text(json.dumps(self.data,indent=2,allow_nan=False)+'\n');tmp.replace(target)
        def warm(self):
            import torch
            owner=self.owner;capacity=owner.seq_lens.numel();assert capacity>=64
            before=torch.cuda.get_rng_state().clone()
            for n in range(1,65):
                binary=kernel.prepare_decode_metadata.warmup(owner.num_computed_tokens,owner.req_indices.gpu,
                    owner.query_pos.gpu,owner.num_scheduled_tokens.gpu,owner.positions,owner.seq_lens,
                    grid=((capacity+255)//256,),N=n,CAPACITY=capacity,BLOCK=256,num_warps=4,num_stages=1)
                assert binary.metadata.num_warps==4 and binary.metadata.num_stages==1
            assert torch.equal(before,torch.cuda.get_rng_state())
            self.data.update(warmup_complete=True,warmup_batches=list(range(1,65)),capacity=capacity,
                             warmup_RNG_state_unchanged=True,original_constructor_called_once=True)
            self.emit(True)
        def selected(self,scheduled,n,total):
            self.data['calls']+=1
            ready=source.eligible(self.owner,scheduled,n,total)
            if ready:
                self.data['eligible_calls']+=1;key=str(n)
                self.data['eligible_by_batch'][key]=self.data['eligible_by_batch'].get(key,0)+1
                if mode=='candidate':
                    self.data['selected_calls']+=1
                    self.data['selected_by_batch'][key]=self.data['selected_by_batch'].get(key,0)+1
            else:self.data['fallback_calls']+=1
            self.emit();return ready and mode=='candidate'
        def launch(self,n):
            binary=kernel.launch(self.owner,n);self.data['fused_launches']+=1
            if str(n) not in self.data['actual_kernels']:
                self.data['actual_kernels'][str(n)]=dict(N=n,CAPACITY=self.owner.seq_lens.numel(),
                    num_warps=binary.metadata.num_warps,num_stages=binary.metadata.num_stages,
                    source_sha256=sha(BASE/'kernel.py'),n_regs=getattr(binary,'n_regs',None),n_spills=getattr(binary,'n_spills',None))
            self.emit();return binary

    def select(owner,scheduler_output,scheduled,n,total):
        controller=getattr(owner,"_tianshu_decode_metadata_controller",None)
        return controller.selected(scheduled,n,total) if controller is not None else False
    def launch(owner,n):return owner._tianshu_decode_metadata_controller.launch(n)
    module.__dict__['_tianshu_decode_metadata_selected']=select
    module.__dict__['_tianshu_decode_metadata_launch']=launch
    namespace={};exec(compile(text,str(generated),'exec'),module.__dict__,namespace)
    cls._prepare_inputs=functools.wraps(original)(namespace['_prepare_inputs'])
    @functools.wraps(original_init)
    def initialize(owner,*args,**kwargs):
        result=original_init(owner,*args,**kwargs)
        controller=Controller(owner);owner._tianshu_decode_metadata_controller=controller
        controller.warm();atexit.register(controller.emit,True)
        return result
    cls.__init__=initialize;cls._tianshu_decode_metadata_installed=True
    assert cls._prepare_inputs.__code__.co_filename==str(generated)
    return binding
