# SPDX-License-Identifier: Apache-2.0
PLANS={
 'K8V16':dict(storage='K8V16',reader='current',pv=False,tile=16,warps=1),
 'V8Native':dict(storage='K8VI8Heads',reader='v8',pv=False,tile=16,warps=1),
 'PVFalse':dict(storage='K8VI8Heads',reader='pv',pv=False,tile=16,warps=1),
 'PV32w2':dict(storage='K8VI8Heads',reader='pv',pv=True,tile=32,warps=2),
 'PV64w2':dict(storage='K8VI8Heads',reader='pv',pv=True,tile=64,warps=2),
 'PV32w4':dict(storage='K8VI8Heads',reader='pv',pv=True,tile=32,warps=4),
 'PV64w4':dict(storage='K8VI8Heads',reader='pv',pv=True,tile=64,warps=4),
}
MODES=tuple(PLANS)
CASES=((1,33),(1,4096),(8,4096),(16,4096),(32,4096),(40,4096),(40,16384),(64,4096),(64,5120),(128,4096))
CURRENT_SHA='f1df731bae6cf48d5f38eef0a0adb5863ade2145667aab1ac976b9d473bdb002'
ALGORITHM_ATOL=.001
ALGORITHM_RTOL=.005
ALGORITHM_RELATIVE_L2=.01
