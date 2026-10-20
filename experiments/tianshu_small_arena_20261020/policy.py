# SPDX-License-Identifier: Apache-2.0
"""Make only the reusable prefill staging capacity smaller; native fallback remains."""
import hashlib,json,os
from pathlib import Path
NATIVE_SHA='c5a3dd7d8190d4bd5c61510e21482a568f161ddd8ca9e1a519f0be0a31ff2555'
CAPACITIES={'control':32768,'candidate':8192}

def install(module,mode,directory):
    assert mode in CAPACITIES and hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==NATIVE_SHA
    original=module.BoundedDensePrefill
    assert not getattr(original,'_small_arena_policy',False)
    pages=CAPACITIES[mode]
    class CapacityBoundedDensePrefill(original):
        _small_arena_policy=True
        def __init__(self):
            super().__init__()
            self.limit_pages=pages
        def reserve(self,device='cuda'):
            if self.storage is not None:
                return
            torch=module.torch
            assert not torch.cuda.is_current_stream_capturing()
            before=torch.cuda.memory_allocated()
            self.storage=torch.full((self.limit_pages+1,2,16,2,128),17.,device=device,dtype=torch.bfloat16)
            self.dense_ids=torch.arange(self.limit_pages,device=device,dtype=torch.int32)
            self.reservation=dict(capacity_pages=self.limit_pages,
                scratch_bytes=self.storage.numel()*self.storage.element_size(),
                table_bytes=self.dense_ids.numel()*self.dense_ids.element_size(),
                allocated_before=before,allocated_after=torch.cuda.memory_allocated(),single_reusable_arena=True)
    assert CapacityBoundedDensePrefill.__call__ is original.__call__
    assert CapacityBoundedDensePrefill.verify_last is original.verify_last
    module.BoundedDensePrefill=CapacityBoundedDensePrefill
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    row=dict(mode=mode,capacity_pages=pages,scratch_bytes=(pages+1)*16384,table_bytes=pages*4,
             native_source=str(module.__file__),native_sha256=NATIVE_SHA,
             original_call_inherited=True,original_gather_decode_unchanged=True,
             original_over_capacity_fallback_unchanged=True,RNG_not_read_or_modified=True,
             policy_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (directory/f'capacity_{os.getpid()}.json').write_text(json.dumps(row,indent=2)+'\n')
    return row
