# SPDX-License-Identifier: Apache-2.0
"""Same strict32K geometry; only standalone initial outlier proposals differ."""
import hashlib,json
from pathlib import Path
PLANS=[('geo',32768,16384,16),('shift_half',32768,16384,16),('shift_one',32768,16384,16)]
class Router:
    def __init__(self,kernel,plan,artifact_root=None,corrected_kernel=None):
        self.kernel,self.plan,self.metadata=kernel,plan,None;self.artifact_root=artifact_root;self.corrected_kernel=corrected_kernel
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            label,tile,trunc,warps=self.plan
            assert kwargs['VOCAB_SIZE']==130560 and kwargs['BATCH_SIZE']>=1
            corrected=not kwargs['TOPK_ENABLED']
            selected=self.corrected_kernel[label] if corrected else self.kernel
            if not corrected:tile,trunc,warps=8192,4096,16
            kwargs.update(BLOCK_SIZE=tile,BLOCK_SIZE_TRUNC=trunc,num_warps=warps,num_stages=1)
            compiled=selected[grid](*args,**kwargs)
            actual=Path(selected.fn.__code__.co_filename)
            self.metadata=dict(label=label,source=str(actual),source_sha256=hashlib.sha256(actual.read_bytes()).hexdigest(),
                corrected_source_active=corrected,BLOCK_SIZE=tile,BLOCK_SIZE_TRUNC=trunc,num_warps=compiled.metadata.num_warps,
                num_stages=compiled.metadata.num_stages,shared=compiled.metadata.shared,grid=list(grid),n_regs=getattr(compiled,'n_regs',None),
                n_spills=getattr(compiled,'n_spills',None),VOCAB_SIZE=kwargs['VOCAB_SIZE'],TOPK_ENABLED=kwargs['TOPK_ENABLED'],TOPP_ENABLED=kwargs['TOPP_ENABLED'])
            assert self.metadata['num_warps']==warps and self.metadata['num_stages']==1
            if self.artifact_root is not None and kwargs['BATCH_SIZE']==64 and corrected:
                root=Path(self.artifact_root)/label
                if not root.exists():
                    root.mkdir(parents=True);files={}
                    for name,value in compiled.asm.items():
                        assert name.replace('_','').isalnum()
                        raw=value.encode() if isinstance(value,str) else value if isinstance(value,bytes) else None
                        if raw is not None:(root/name).write_bytes(raw);files[name]=hashlib.sha256(raw).hexdigest()
                    (root/'manifest.json').write_text(json.dumps(dict(metadata=self.metadata,files=files),indent=2)+'\n')
            return compiled
        return launch
