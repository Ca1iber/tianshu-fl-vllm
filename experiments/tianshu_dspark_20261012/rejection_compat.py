# SPDX-License-Identifier: Apache-2.0
"""CoreX-compatible positive FP32 uniform thresholds; rejection ratio unchanged."""
import json
import os
from pathlib import Path
import torch


def install(module):
    recorded=False
    def generate_uniform_probs(num_tokens,num_draft_tokens,generators,device):
        nonlocal recorded
        uniform=torch.rand((num_tokens,),dtype=torch.float32,device=device)
        start=0
        for req_idx,count in enumerate(num_draft_tokens):
            if count==0:continue
            end=start+count
            generator=generators.get(req_idx)
            if generator is not None:uniform[start:end].uniform_(generator=generator)
            start=end
        # Torch FP32 rand is in[0,1). Reflect to(0,1], preventing a zero
        # threshold from accepting a proposal with zero target probability.
        uniform=1.-uniform
        if num_tokens and not recorded:
            recorded=True
            host=uniform.cpu()
            root=Path(os.environ['TIANSHU_DSPARK_RECORD_DIR'])/'rejection_threshold'
            root.mkdir(parents=True,exist_ok=True)
            (root/f'route_{os.getpid()}.json').write_text(json.dumps({
                'dtype':str(uniform.dtype),'num_tokens':num_tokens,'minimum':host.min().item(),
                'maximum':host.max().item(),'unique':host.unique().numel(),
                'interval':'(0,1]','rejection_kernel_modified':False,
                'synthetic_acceptance_enabled':False,'probability_ratio_changed':False,
                'scope':'FP32 uniform threshold generation compatibility only; request generators retained'},indent=2)+'\n')
        return uniform
    module.generate_uniform_probs=generate_uniform_probs
