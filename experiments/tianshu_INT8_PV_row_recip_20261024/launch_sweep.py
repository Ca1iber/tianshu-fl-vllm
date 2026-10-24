# SPDX-License-Identifier: Apache-2.0
"""Same current control, scaled floating PV controls, and integer PV candidates."""
import importlib.util
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parent
path=BASE.parent/'tianshu_V_e5m2_launch_sweep_20261024/launch_sweep.py'
spec=importlib.util.spec_from_file_location('_PV_compiled_recorder',path)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
CompiledRecorder=module.CompiledRecorder
CONFIGS=[
 dict(label="STRICT_INT8_S4_T32_W1",segments=4,TILE_SIZE=32,num_warps=1,PV_INT8=True,variant="strict"),
 dict(label="ROW_RECIP_INT8_S4_T32_W1",segments=4,TILE_SIZE=32,num_warps=1,PV_INT8=True,variant="row_recip")]

class FlagKernel:
    def __init__(self,wrapped,flag):
        self.wrapped,self.flag=wrapped,flag
    def __getattr__(self,name):return getattr(self.wrapped,name)
    def __getitem__(self,grid):
        launch=self.wrapped[grid]
        def invoke(*args,**kw):
            return launch(*args,**kw,PV_INT8=self.flag)
        return invoke

def attach(model,component_launch,output):
    import torch,triton
    ordinary,producer=component_launch.attach(model['reader'])
    if model['label']!='control':
        producer.kernel=FlagKernel(producer.kernel,model['config']['PV_INT8'])
    target=triton.runtime.driver.active.get_current_target()
    recorder=CompiledRecorder(producer.kernel,output,model['label'],
        dict(backend=target.backend,arch=target.arch,warp_size=target.warp_size))
    producer.kernel=recorder;producer._sweep_codegen=recorder
    if model['label']=='control':return ordinary,producer
    config=dict(model['config']);assert config in CONFIGS
    def invoke(kwargs):
        kw=dict(kwargs)
        assert kw['q'].dtype==torch.bfloat16 and kw['k'].dtype==torch.uint8 and kw['v'].dtype==torch.int8
        assert kw['kv_quant_mode'].value==2
        assert kw['k'].stride(-1)==kw['v'].stride(-1)==1
        assert kw['max_seqlen_q']==1 and kw['seqused_k'].numel()==64
        producer.pure_decode=True
        producer.decode_config={k:config[k] for k in ('segments','TILE_SIZE','num_warps')}
        kw['num_par_softmax_segments']=config['segments']
        answer=model['reader'].unified_attention(**kw)
        assert producer.last_record['IS_3D']
        assert producer.last_record['segments']==config['segments'] and producer.last_record['tile']==config['TILE_SIZE']
        assert producer.last_record['warps']==config['num_warps']
        producer.last_record['address_selection']=dict(recorder.last_selection)
        producer.last_record['PV_INT8']=config['PV_INT8']
        producer.last_record['KV_QUANT_MODE']=2
        return answer
    return invoke,producer
