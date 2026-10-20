# SPDX-License-Identifier: Apache-2.0
"""Runtime native cache strides are checked on the host before paired V load."""
def check_pair(strides,head,padded,dtype,mode):
    assert mode==2 and head==padded==128 and dtype=='torch.int8'
    assert len(strides)==4 and all(type(v) is int for v in strides)
    assert strides[3]==1 and all(v>0 and v%2==0 for v in strides[:3])
    return True
