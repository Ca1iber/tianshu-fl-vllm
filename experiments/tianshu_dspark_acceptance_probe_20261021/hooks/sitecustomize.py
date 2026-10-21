# SPDX-License-Identifier: Apache-2.0
"""Reuse qualified legacy compatibility; install only explicit DSpark length fix."""
import importlib.abc,importlib.machinery,importlib.util,json,os,runpy,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
POLICY=BASE.parent/'tianshu_dspark_input_contract_20261021/policy.py'
def install(module):
    spec=importlib.util.spec_from_file_location('_tianshu_dspark_seven_slot_policy',POLICY)
    policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy);policy.install(module)
    selected=module.ModelRunnerFL._input_fits_in_drafter;seen=set()
    def recorded(self,cad):
        result=selected(self,cad)
        if cad is not None and policy.is_dspark(self.speculative_config):
            ready=Path(os.environ['TIANSHU_DSPARK_ACCEPTANCE_READY_FILE']).exists()
            key='request' if ready else 'startup'
            if key not in seen:
                seen.add(key);draft=self.drafter
                assert type(draft).__name__=='DSparkProposer' and draft.backbone_block_size==7
                target_head=self.model.lm_head.weight;draft_head=draft.model.lm_head.weight
                row=dict(pid=os.getpid(),phase=key,physical_query_width=7,target_proposals=self.num_spec_tokens,
                    max_seq_len=cad.max_seq_len,effective_drafter_max_model_len=self.effective_drafter_max_model_len,
                    accepted_input=result,method=self.speculative_config.method,
                    draft_sample_method=self.speculative_config.draft_sample_method,
                    rejection_sample_method=self.speculative_config.rejection_sample_method,
                    drafter_type=type(draft).__name__,model_runner_source=module.__file__,policy_source=str(POLICY),
                    target_layers=len(self.model.model.layers),draft_layers=len(draft.model.model.layers),
                    target_head_dtype=str(target_head.dtype),draft_head_dtype=str(draft_head.dtype),
                    lm_head_shared=target_head.data_ptr()==draft_head.data_ptr(),
                    auxiliary_layers=list(self.model.model.aux_hidden_state_layers))
                root=Path(os.environ['TIANSHU_DSPARK_RECORD_DIR'])/'seven_slot_input_fit';root.mkdir(parents=True,exist_ok=True)
                (root/f'{os.getpid()}_{key}.json').write_text(json.dumps(row,indent=2)+'\n')
        return result
    module.ModelRunnerFL._input_fits_in_drafter=recorded
class Loader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):self.original.exec_module(module);install(module)
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!='vllm_fl.worker.model_runner':return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path);assert spec is not None and spec.loader is not None
        spec.loader=Loader(spec.loader);return spec
if 'vllm_fl.worker.model_runner' in sys.modules:install(sys.modules['vllm_fl.worker.model_runner'])
else:sys.meta_path.insert(0,Finder())
runpy.run_path(str(BASE.parent/'tianshu_dspark_20261012/hooks/sitecustomize.py'),run_name='_tianshu_legacy_dspark_hooks')
