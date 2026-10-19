# SPDX-License-Identifier: Apache-2.0
"""Check actual helper control flow; actual Torch/FlagGems RNG still needs GPU proof."""
import ast,hashlib,json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from warm import warm_exponential


class FakeTorch:
    float32='FP32'
    def __init__(self,fail_batch=None):
        self.state=[123,456];self.calls=[];self.fail_batch=fail_batch;self.capturing=False
        self.cuda=SimpleNamespace(is_current_stream_capturing=lambda:self.capturing,
            current_device=lambda:0,synchronize=lambda device:None)
        self.random=SimpleNamespace(fork_rng=self.fork)
    @contextmanager
    def fork(self,devices,enabled):
        assert devices==[0] and enabled
        previous=list(self.state)
        try:yield
        finally:self.state[:]=previous
    def device(self,name):return SimpleNamespace(index=0)
    def empty(self,shape,dtype,device):
        assert shape==(64,130560) and dtype==self.float32 and device=='cuda:0'
        return Tensor(self,64)


class Tensor:
    def __init__(self,owner,batch):self.owner,self.batch=owner,batch
    def __getitem__(self,s):
        assert s.start is s.step is None and 1<=s.stop<=64
        return Tensor(self.owner,s.stop)
    def numel(self):return self.batch*130560
    def is_contiguous(self):return True
    def exponential_(self):
        self.owner.calls.append(self.batch)
        self.owner.state[0]+=3;self.owner.state[1]+=self.numel()//8
        if self.batch==self.owner.fail_batch:raise RuntimeError('CPU sentinel')
        return self


t=FakeTorch();r=warm_exponential(t)
assert t.calls==list(range(64,0,-1)) and t.state==[123,456]
assert r['complete'] and r['maximum_noise_bytes']==33423360
assert [s['elements'] for s in r['shapes']]==[n*130560 for n in range(64,0,-1)]
t=FakeTorch(fail_batch=17)
try:warm_exponential(t)
except RuntimeError as e:assert str(e)=='CPU sentinel'
else:raise AssertionError('exception swallowed')
assert t.calls==list(range(64,16,-1)) and t.state==[123,456]
t=FakeTorch();t.capturing=True
try:warm_exponential(t)
except AssertionError:pass
else:raise AssertionError('graph capture accepted')
assert not t.calls
for kw in (dict(max_batch=63),dict(vocab=1)):
    try:warm_exponential(FakeTorch(),**kw)
    except AssertionError:pass
    else:raise AssertionError('unsupported geometry accepted')
b=Path(__file__).resolve().parent
for p in b.glob('*.py'):ast.parse(p.read_text())
g=dict(complete=True,CPU_only=True,actual_helper_all64_shape_control_flow_checked=True,
    fork_context_used_on_success_and_exception=True,graph_and_geometry_guards_checked=True,
    real_Torch_and_FlagGems_RNG_not_yet_verified=True,GPU_not_run=True,adopted=False,
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(b.glob('*.py'))})
(b/'CPU_GATE_32g.json').write_text(json.dumps(g,indent=2)+'\n');print(json.dumps(g))
