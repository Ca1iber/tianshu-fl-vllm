# SPDX-License-Identifier: Apache-2.0
"""Verify exact physical address bijections and inverse maps before GPU execution."""
import argparse,hashlib,json
from pathlib import Path
from source import CHUNKS,k_local,render_attention,render_writer
p=argparse.ArgumentParser();p.add_argument('--attention',type=Path,required=True);p.add_argument('--writer',type=Path,required=True);a=p.parse_args()
records=[]
for x in CHUNKS:
    addresses={}
    for t in range(16):
        for h in range(2):
            for d in range(128):
                i=k_local(t,h,d,x);assert i not in addresses;addresses[i]=(t,h,d)
                assert i//2048==h
                local=i%2048;chunk=local//(16*x);slot=(local//x)%16;inner=local%x
                assert (slot,h,chunk*x+inner)==(t,h,d)
    assert set(addresses)==set(range(4096))
    # K stays within its half of a page, including strided/noncontiguous pages.
    for stride in (8192,16384):
        for block in (0,1,17):
            assert max(block*stride+i for i in addresses)<block*stride+4096
            assert not {block*stride+i for i in addresses}&set(range(block*stride+4096,block*stride+8192))
    at=render_attention(a.attention.read_bytes(),x);wr=render_writer(a.writer.read_bytes(),x)
    records.append(dict(x=x,addresses=4096,inverse_exact=True,attention_sha256=hashlib.sha256(at.encode()).hexdigest(),writer_sha256=hashlib.sha256(wr.encode()).hexdigest()))
base=Path(__file__).resolve().parent
r=dict(complete=True,CPU_only=True,GPU_measured=False,physical_address_cases=4*4096,records=records,
 sources={n:hashlib.sha256((base/n).read_bytes()).hexdigest() for n in ('source.py','check_cpu.py','probe_layout.py','run_probe.py')},
 scope='Only source transforms and exact page-address bijections; runtime geometry guard, GPU writer, reader and graph tests still required')
(base/'CPU_GATE_32g.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
