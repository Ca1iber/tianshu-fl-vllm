# SPDX-License-Identifier: Apache-2.0
import hashlib,json
from pathlib import Path
p=Path('/usr/local/lib/python3.12/site-packages/flag_gems/ops/argmax.py');raw=p.read_text();assert hashlib.sha256(p.read_bytes()).hexdigest()=='646a6e1cf1a8b1b9d24fb772b3e09e1f7bdea2f0449167ba9abfafaaf23576c8'
start=raw.index('def argmax_kernel_inner(');end=raw.index('\n\ndef argmax(',start);body=raw[start:end]
assert body.count('    inp,\n')==1;body=body.replace('    inp,\n','    inp,\n    noise,\n',1)
assert body.count('inp_vals = tl.load(')==3
lines=body.splitlines();out=[]
for line in lines:
 out.append(line)
 if 'inp_vals = tl.load(' in line:
  indent=line[:len(line)-len(line.lstrip())];masked='mask=mask, other=1.0' if 'mask=mask' in line else ''
  args=(', '+masked) if masked else ''
  out.extend([indent+'noise_vals = tl.load(noise + offset'+args+')',indent+'inp_vals = inp_vals / noise_vals',indent+'tl.store(inp_ptrs, inp_vals'+(', mask=mask' if masked else '')+')'])
text=raw[:start]+'\n'.join(out)+raw[end:];base=Path(__file__).resolve().parent;target=base/'generated_argmax.py';target.write_text(text)
proof=dict(parent=str(p),parent_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),generated_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),changes='one Q parameter; at each of the three original load sites load Q, apply original FP32 /, write scores back; all native reduction/tie/NaN/minimum/loop/heuristics code unchanged',original_file_modified=False)
(base/'SOURCE_PROOF_32g.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
