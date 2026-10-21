# SPDX-License-Identifier: Apache-2.0
CASES=((8,False),(8,True),(64,False),(64,True))
def configurations(m):
    assert m in (8,64)
    return [dict(bm=32 if m==8 else 64,bn=bn,bk=128,warps=warps,stages=1,mode=mode)
            for mode in ('two_dot','single_dot') for bn in (64,128) for warps in (2,4)]
def baseline(m):
    assert m in (8,64)
    return dict(bm=32 if m==8 else 64,bn=256,bk=128,warps=4,stages=1)
