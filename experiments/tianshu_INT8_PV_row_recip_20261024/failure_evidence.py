# SPDX-License-Identifier: Apache-2.0
"""Save numerical failure before asserting; fulloutputs plus up to8 failing head inputs."""
import hashlib,json
import numpy as np

def save(root,name,label,phase,q,k,v,used,actual,expected):
    a=actual.float().cpu().numpy();e=expected.float().cpu().numpy()
    bad=np.abs(a-e)>.002+.01*np.abs(e)
    positions=np.argwhere(bad)
    heads=sorted({(int(row[0]),int(row[1])) for row in positions})
    arrays=dict(actual_BF16_words=actual.contiguous().view(__import__('torch').uint16).cpu().numpy(),
        expected_FP32=e,lengths=used.cpu().numpy(),bad_indices=positions)
    selected=heads[:8]
    for index,(sequence,head) in enumerate(selected):
        length=int(arrays['lengths'][sequence])
        arrays[f'Q{index}']=q[sequence,head].float().cpu().numpy()
        arrays[f'K{index}']=k[sequence,:length,head//8].float().cpu().numpy()
        arrays[f'V{index}']=v[sequence,:length,head//8].float().cpu().numpy()
    path=root/f'FAILED_{name}_{label}_phase{phase}.npz'
    np.savez_compressed(path,**arrays)
    record=dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        failed_elements=int(bad.sum()),full_output_and_FP32_reference_saved=True,
        selected_failing_heads=selected,total_failing_heads=len(heads),failed_head_input_limit=8,
        scope='Full failed-element recount against actual GPU FP32 reference; selected-head CPU FP64 diagnostic possible')
    path.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    return record
