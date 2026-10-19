# SPDX-License-Identifier: Apache-2.0
"""More independent row programs, with separately owned sufficiently large scratch."""
import hashlib
import json
from pathlib import Path

PLANS = [
    ('adopted',8192,4096,16,1),('identity',8192,4096,16,1),
    ('fixed32_geo',32768,16384,16,1),('fixed32_grid2',32768,16384,16,2),
    ('fixed32_grid4',32768,16384,16,4),('fixed16_geo',16384,8192,16,1),
    ('fixed16_grid2',16384,8192,16,2),('fixed16_grid4',16384,8192,16,4),
]
# Graphs share this process-owned scratch only on the same sequential stream.
# Keep storage alive for all captures; never rely on a temporary Router lifetime.
SCRATCH = {}


class Router:
    def __init__(self, kernel, plan, artifact_root=None, corrected_kernel=None):
        self.kernel, self.plan, self.metadata = kernel, plan, None
        self.artifact_root, self.corrected_kernel = artifact_root, corrected_kernel

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            label,tile,trunc,warps,multiplier = self.plan
            batch,vocab = kwargs['BATCH_SIZE'],kwargs['VOCAB_SIZE']
            assert vocab == 130560 and batch >= 8
            active = label.startswith('fixed') and not kwargs['TOPK_ENABLED']
            selected = self.corrected_kernel['geo'] if active else self.kernel
            original_grid = int(grid[0]); launch_grid = original_grid
            arguments = list(args); original_buffer_rows = args[2].shape[0]
            if active:
                launch_grid = min(batch, original_grid*multiplier)
                assert 1 <= original_grid <= launch_grid <= batch
                if launch_grid > original_grid:
                    logits = args[0]
                    key = (logits.device, logits.dtype, batch, vocab)
                    scratch = SCRATCH.get(key)
                    if scratch is None:
                        scratch = logits.new_empty((batch,vocab))
                        SCRATCH[key] = scratch
                    arguments[2] = scratch
                assert arguments[2].shape[0] >= launch_grid
            elif label.startswith('fixed'):
                tile,trunc,warps = 8192,4096,16
            kwargs.update(BLOCK_SIZE=tile,BLOCK_SIZE_TRUNC=trunc,num_warps=warps,num_stages=1)
            compiled = selected[(launch_grid,)](*arguments,**kwargs)
            self.metadata = dict(label=label,corrected_source_active=active,
                source_module=selected.fn.__module__,BLOCK_SIZE=tile,BLOCK_SIZE_TRUNC=trunc,
                num_warps=compiled.metadata.num_warps,num_stages=compiled.metadata.num_stages,
                shared=compiled.metadata.shared,grid=[launch_grid],original_grid=[original_grid],
                requested_grid_multiplier=multiplier,original_buffer_rows=original_buffer_rows,
                actual_buffer_rows=arguments[2].shape[0],scratch_owned=launch_grid>original_grid,
                n_regs=getattr(compiled,'n_regs',None),n_spills=getattr(compiled,'n_spills',None),
                VOCAB_SIZE=vocab,TOPK_ENABLED=kwargs['TOPK_ENABLED'],TOPP_ENABLED=kwargs['TOPP_ENABLED'])
            assert self.metadata['num_warps'] == warps and self.metadata['num_stages'] == 1
            if self.artifact_root is not None and batch == 64 and not kwargs['TOPK_ENABLED']:
                root = Path(self.artifact_root)/label
                if not root.exists():
                    root.mkdir(parents=True); files = {}
                    for name,value in compiled.asm.items():
                        assert name.replace('_','').isalnum()
                        raw = value.encode() if isinstance(value,str) else value if isinstance(value,bytes) else None
                        if raw is not None:
                            (root/name).write_bytes(raw); files[name] = hashlib.sha256(raw).hexdigest()
                    (root/'manifest.json').write_text(json.dumps(dict(metadata=self.metadata,files=files),indent=2)+'\n')
            return compiled
        return launch
