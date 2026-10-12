# SPDX-License-Identifier: Apache-2.0
"""Compare the actual FlagGems sampling-row gather with CPU and a prefix view."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import signal
import subprocess


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output;assert not root.exists();root.mkdir(parents=True)
    assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
    lease=Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json')
    remaining=int((datetime.fromisoformat(json.loads(lease.read_text())['stop_at_utc'])-
                   datetime.now(timezone.utc)).total_seconds())
    assert remaining>0
    signal.alarm(min(900,remaining))
    import torch
    from flag_gems.ops.index import index
    torch.manual_seed(2026101209)
    rows=[]
    for batch in (1,2,3,4,5,6,7,8,16,64):
        count=batch*7
        for strided in (False,True):
            storage=torch.randn((count+8)*(2 if strided else 1),2048,
                                dtype=torch.bfloat16,device='cuda')
            source=storage[::2] if strided else storage
            source[count:].fill_(17.)
            initial=source.cpu()
            indices=torch.arange(count,device='cuda',dtype=torch.int32)
            gathered=index(source,[indices])
            view=source[:count]
            expected=initial[:count]
            entry={'batch':batch,'query_rows':count,'input_strides':list(source.stride()),
                   'gather_matches_cpu_bitwise':torch.equal(gathered.cpu(),expected),
                   'view_matches_cpu_bitwise':torch.equal(view.cpu(),expected),
                   'gather_and_view_bitwise':torch.equal(gathered.cpu(),view.cpu()),
                   'source_unchanged':torch.equal(source.cpu(),initial),
                   'view_shares_pointer':view.data_ptr()==source.data_ptr()}
            rows.append(entry)
            (root/'result.json').write_text(json.dumps({'complete':False,'cases':rows},indent=2))
    source=Path(index.__code__.co_filename)
    data={'complete':True,'official_benchmark':False,'accuracy_test':False,
          'index_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'cases':rows,'passed':all(all(x[k] for k in ('gather_matches_cpu_bitwise',
              'view_matches_cpu_bitwise','gather_and_view_bitwise','source_unchanged','view_shares_pointer')) for x in rows)}
    (root/'result.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))
    assert data['passed']


if __name__=='__main__':main()
