# SPDX-License-Identifier: Apache-2.0

def configuration(m,selected):
    assert isinstance(m,int) and 128<=m<=4096 and m%128==0
    reference=min(size for size in selected if size>=m)
    return reference,dict(selected[reference])

def eligible(mode,active,x,warm):
    return bool(mode=='candidate' and active and x.ndim==2 and x.shape[0] in warm and x.shape[1]==2048 and x.is_contiguous())
